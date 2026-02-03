import pygame
import collections
import math
import os
import csv
import asyncio

# --- Imports ---
from entity import Vessel, Francisco
from scenario_generator import (head_on_scenario, cross_over_scenario,
                                over_taking_scenario, multi_vessel_scenario,
                                multi_vessel_scenario_2, traffic_separation_scenario)
from response_parser import Maneuver, parse_llm_response_for_all

from llm_text_manager import LLMTextManager
from llm_vision_manager import LLMVisionManager

# --- IMPORT ALL PROMPT GENERATORS ---
import prompts_generator.minimal_prompt as minimal_gen
import prompts_generator.moderate_prompt as moderate_gen
import prompts_generator.detailed_prompt as detailed_gen
import prompts_generator.natural_language_prompt as natural_gen
import prompts_generator.tss_prompt as tss_gen
import prompts_generator.prompt_generator as standard_gen


class EntityManager:
    def __init__(self, game_manager, mode='rag', llm_provider='openai', prompt_type='minimal'):
        self.game_manager = game_manager
        self.mode = mode
        self.llm_provider = llm_provider
        self.prompt_type = prompt_type

        self.vessels = []
        self.movement_active = False
        self.goal_queue = []
        self.current_scenario = None
        self._sim_time = 0.0

        self.llm_cooldown = {}
        self.llm_cooldown_duration = 5.0
        self.pixels_per_km = 1000.0
        self.radar_range_km = 0.8
        self.llm_call_count = 0

        # --- HISTORY INITIALIZATION  ---
        self.max_history_length = 3
        self.history_vessel_data = collections.deque(maxlen=self.max_history_length)
        self.history_responses = collections.deque(maxlen=self.max_history_length)
        # ------------------------------------------------------------

        if self.mode == 'rag':
            self.decision_maker = LLMTextManager(provider=llm_provider)
            self.is_vision_based = False
            self.use_rag = True
        elif self.mode == 'standard':
            # Standard mode uses the Text Manager but will bypass RAG
            self.decision_maker = LLMTextManager(provider=llm_provider)
            self.is_vision_based = False
            self.use_rag = False
        else:
            # 'vision' uses the vision manager
            self.decision_maker = LLMVisionManager(provider=llm_provider)
            self.is_vision_based = True
            self.use_rag = False  # Not applicable

        self.take_screenshot_on_next_render = False
        self.screenshot_dir = f"screenshots_{mode}"
        if not os.path.exists(self.screenshot_dir): os.makedirs(self.screenshot_dir)

        self.llm_log_path = f"logs/llm_{mode}_interactions_log.csv"
        self.llm_log_fieldnames = ['llm_call_id', 'simulation_time_s', 'involved_vessels', 'prompt_data',
                                   'llm_response_json']
        os.makedirs(os.path.dirname(self.llm_log_path), exist_ok=True)
        if not os.path.exists(self.llm_log_path):
            with open(self.llm_log_path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=self.llm_log_fieldnames).writeheader()

        self.color_map = {(255, 0, 0): "Red", (0, 0, 255): "Blue", (0, 255, 0): "Green", (255, 165, 0): "Orange",
                          (128, 0, 128): "Purple"}

    def load_scenario(self, selected_option):
        print(f"--- Clearing vessels and history for new scenario: {selected_option} ---")
        self.vessels.clear()

        # Clear history safely
        self.history_vessel_data.clear()
        self.history_responses.clear()

        self._sim_time = 0.0
        self.llm_call_count = 0

        sw, sh = self.game_manager.graphics_manager.width, self.game_manager.graphics_manager.height
        scenarios = {
            "Head-On Scenario": ("head_on", head_on_scenario),
            "Cross Over Scenario": ("cross_over", cross_over_scenario),
            "Over Taking Scenario": ("over_taking", over_taking_scenario),
            "Multi vessel Scenario": ("multi_vessel", multi_vessel_scenario),
            "Multi vessel Scenario2": ("multi_vessel_2", multi_vessel_scenario_2),
            "Traffic Separation Scenario": ("tss", traffic_separation_scenario)
        }

        if selected_option in scenarios:
            self.current_scenario, func = scenarios[selected_option]
            func(self, sw, sh)
            self.movement_active = True
        else:
            print(f"Warning: Unknown scenario '{selected_option}'")

    # Single, async update method
    async def update_vessels(self, dt):
        if not self.movement_active and not self.goal_queue: return
        self._sim_time += dt
        if dt <= 0: return

        if self.goal_queue:
            for entry in self.goal_queue:
                if isinstance(entry, tuple) and len(entry) == 2:
                    v, g = entry
                    if isinstance(v, int): v = next((x for x in self.vessels if id(x) == v), None)
                    if v: v.goal = g
            self.goal_queue.clear()
        if not self.movement_active: return

        # 2. Execute
        any_move = False
        for v in self.vessels:
            is_tss = (self.current_scenario == 'tss' and isinstance(v, Francisco))
            v.update_heading_and_speed(dt, is_tss)
            v.update_position(dt)
            if v.goal is None and not v.in_maneuver and v.speed < 0.1: v.speed = 0
            if v.goal or v.speed > 0.1: any_move = True
        self.movement_active = any_move

        # 3. Predict
        for v in self.vessels: v.calculate_desired_velocity()
        to_query = []
        MOVING_THRESHOLD = 0.1
        for v in self.vessels:
            if (v.speed / self.pixels_per_km * 3600) <= MOVING_THRESHOLD:
                v.in_maneuver = False;
                continue
            conflict_found = False
            for other in self.vessels:
                if v is other: continue
                if (other.speed / self.pixels_per_km * 3600) <= MOVING_THRESHOLD: continue
                if (math.hypot(v.x - other.x, v.y - other.y) / self.pixels_per_km) > self.radar_range_km: continue
                if self.predict_collision(v, other):
                    conflict_found = True;
                    break

            if conflict_found:
                v.in_maneuver = True
                should_query = False
                if self.current_scenario == 'tss':
                    if isinstance(v, Francisco): should_query = True
                else:
                    should_query = True

                if should_query and (self._sim_time - self.llm_cooldown.get(id(v), 0.0) >= self.llm_cooldown_duration):
                    to_query.append(v)
            else:
                last = self.llm_cooldown.get(id(v), -999)
                if self._sim_time - last > (self.llm_cooldown_duration + 5.0):
                    v.in_maneuver = False;
                    v.current_maneuver = None

        # 4. AI Decision
        if to_query:
            await self._handle_ai_decision(to_query)

    async def _handle_ai_decision(self, to_query):
        for v in to_query: self.llm_cooldown[id(v)] = self._sim_time
        context_vessels = set(to_query)
        for v in to_query:
            for other in self.vessels:
                if v is not other and self.predict_collision(v, other):
                    context_vessels.add(other)

        current_states = [{
            "id": id(v),
            "pos": (f"{v.x:.1f}", f"{v.y:.1f}"),
            "heading_deg": f"{math.degrees(v.heading):.1f}",
            "speed_kmh": f"{(v.speed / self.pixels_per_km * 3600):.1f}"
        } for v in context_vessels]

        prompt_history = list(self.history_vessel_data)
        resp_history = list(self.history_responses)

        print(f"===== Generating Prompt ({self.prompt_type}) =====")

        prompt = ""

        if self.prompt_type == 'minimal':
            prompt = minimal_gen.generate_vessel_prompt(to_query, list(context_vessels), self.pixels_per_km)
        elif self.prompt_type == 'moderate':
            prompt = moderate_gen.generate_vessel_prompt(to_query, list(context_vessels), self.pixels_per_km)
        elif self.prompt_type == 'detailed':
            prompt = detailed_gen.generate_vessel_prompt(to_query, list(context_vessels), self.pixels_per_km)
        elif self.prompt_type == 'natural':
            prompt = natural_gen.generate_natural_language_prompt(to_query, list(context_vessels), self.pixels_per_km,
                                                                  prompt_history, resp_history)
        elif self.prompt_type == 'tss':
            prompt = tss_gen.generate_tss_crossing_prompt(to_query, list(context_vessels), self.pixels_per_km,
                                                          prompt_history, resp_history)
        else:
            prompt = standard_gen.generate_vessel_prompt(to_query, list(context_vessels), self.pixels_per_km,
                                                         prompt_history, resp_history)

        print(f"===== Calling {self.llm_provider} ({'Vision' if self.is_vision_based else 'Text'})... =====")

        if self.is_vision_based:
            raw = await self.decision_maker.get_llm_decision_from_image(self.game_manager.graphics_manager.screen,
                                                                        prompt)
            self.take_screenshot_on_next_render = True
        else:
            # Text-Based Logic (RAG or Standard)
            if self.use_rag:
                # Call the RAG method
                raw = self.decision_maker.get_llm_decision(prompt)
            else:
                # Call the Standard method (direct LLM call)
                raw = self.decision_maker.get_llm_decision_standard(prompt)

        self._last_response = raw
        self.llm_call_count += 1
        print("Raw LLM Response:", raw)

        if raw:
            self._log_interaction(to_query, prompt, raw)
            parsed = parse_llm_response_for_all(raw)

            if parsed:
                self.history_vessel_data.append(current_states)
                self.history_responses.append(parsed)
                print("\n--- FINAL PARSED ACTIONS ---")
                for entry in parsed:
                    v_update = next((v for v in self.vessels if id(v) == entry['id']), None)
                    if v_update:
                        v_update.set_maneuver(entry['maneuver'])
                        c_name = self.color_map.get(v_update.color, "Unknown")
                        print(f"  > {c_name} Vessel (ID: {entry['id']}): -> {entry['maneuver'].name}")
                print("--------------------------\n")

    def _log_interaction(self, vessels, prompt, response):
        v_str = ", ".join([f"{self.color_map.get(v.color, '?')}({id(v)})" for v in vessels])
        entry = {
            'llm_call_id': self.llm_call_count, 'simulation_time_s': f"{self._sim_time:.2f}",
            'involved_vessels': v_str, 'prompt_data': prompt, 'llm_response_json': response
        }
        try:
            with open(self.llm_log_path, 'a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=self.llm_log_fieldnames).writerow(entry)
        except Exception as e:
            print(f"Log Error: {e}")

    def predict_collision(self, v1, v2, time_horizon_sec=60.0, min_dist_km=0.1):
        p_rel_x = v1.x - v2.x;
        p_rel_y = v1.y - v2.y
        v_rel_x = v1.desired_velocity[0] - v2.desired_velocity[0];
        v_rel_y = v1.desired_velocity[1] - v2.desired_velocity[1]
        v_rel_mag_sq = v_rel_x ** 2 + v_rel_y ** 2
        if v_rel_mag_sq == 0: return False
        dot_product = (p_rel_x * v_rel_x) + (p_rel_y * v_rel_y)
        tcpa = -dot_product / v_rel_mag_sq
        if not (0 < tcpa < time_horizon_sec): return False
        future_x1 = v1.x + v1.desired_velocity[0] * tcpa;
        future_y1 = v1.y + v1.desired_velocity[1] * tcpa
        future_x2 = v2.x + v2.desired_velocity[0] * tcpa;
        future_y2 = v2.y + v2.desired_velocity[1] * tcpa
        dist_sq = (future_x1 - future_x2) ** 2 + (future_y1 - future_y2) ** 2
        dcpa_km = math.sqrt(dist_sq) / self.pixels_per_km
        if dcpa_km < min_dist_km: return True
        return False

    def draw(self, screen):
        """Draws only vessels. Status box removed."""
        for v in self.vessels:
            v.draw(screen)

    def add_vessel(self, v):
        self.vessels.append(v)