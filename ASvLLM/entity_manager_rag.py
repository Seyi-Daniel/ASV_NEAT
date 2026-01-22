import pygame
import time
import collections
import math
import json
import os
import csv
# --- Imports ---
from entity import Vessel, font, Francisco  # Import what's needed
from scenario_generator import (head_on_scenario, cross_over_scenario,
                                over_taking_scenario, multi_vessel_scenario,
                                multi_vessel_scenario_2, traffic_separation_scenario)
# --- Import NEW Text Manager ---
from llm_text_manager import LLMTextManager
# --- Import prompt generators ---
from prompts_generator.prompt_generator import generate_vessel_prompt
from prompts_generator.tss_prompt import generate_tss_crossing_prompt
from response_parser import Maneuver, parse_llm_response_for_all


class EntityManager:
    """
    Manages all entities for the TEXT-ONLY RAG simulation mode.
    """

    def __init__(self, game_manager, llm_provider='openai'):
        self.game_manager = game_manager
        self.vessels = []
        self.movement_active = False
        self.goal_queue = []

        self.llm_cooldown = {}
        self.llm_cooldown_duration = 5.0
        self.pixels_per_km = 1000.0
        self.radar_range_km = 0.8
        self._sim_time = 0.0
        self.llm_call_count = 0
        self._last_response = None

        self.max_history_length = 3
        self.history_vessel_data = collections.deque(maxlen=self.max_history_length)
        self.history_responses = collections.deque(maxlen=self.max_history_length)

        # --- Use new LLMTextManager ---
        self.rag_system = LLMTextManager(provider=llm_provider)
        self.current_scenario = None

        self.color_map = {
            (255, 0, 0): "Red", (0, 0, 255): "Blue", (0, 255, 0): "Green",
            (255, 165, 0): "Orange", (128, 0, 128): "Purple",
        }

        # --- NO SCREENSHOTS in text-only RAG mode ---
        # self.take_screenshot_on_next_render = False
        # self.screenshot_dir = "screenshots_rag"

        self.llm_log_path = "llm_rag_interactions_log.csv"
        self.llm_log_fieldnames = ['llm_call_id', 'simulation_time_s', 'involved_vessels', 'prompt_data',
                                   'llm_response_json']
        if not os.path.exists(self.llm_log_path):
            with open(self.llm_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.llm_log_fieldnames)
                writer.writeheader()

    def draw(self, screen):
        """Draws all vessels and the status box."""
        for vessel in self.vessels:
            vessel.draw(screen)

        if not self.vessels: return
        # Draw status box
        box_width = 350;
        box_height = 40 + (len(self.vessels) * 25);
        box_padding = 10
        box_x = screen.get_width() - box_width - box_padding;
        box_y = box_padding
        pygame.draw.rect(screen, (220, 220, 220), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(screen, (0, 0, 0), (box_x, box_y, box_width, box_height), 2)
        title_surface = font.render("Vessel Actions (RAG Mode)", True, (0, 0, 0))
        screen.blit(title_surface, (box_x + box_padding, box_y + box_padding))
        for i, vessel in enumerate(self.vessels):
            maneuver = vessel.current_maneuver or Maneuver.MAINTAIN_COURSE_SPEED
            maneuver_name = maneuver.name
            color_name = self.color_map.get(vessel.color, "Unknown Color")
            status_text = f"{color_name} Vessel: {maneuver_name}"
            text_surface = font.render(status_text, True, vessel.color)
            text_y = box_y + 40 + (i * 25)
            screen.blit(text_surface, (box_x + box_padding, text_y))

    def add_vessel(self, vessel):
        self.vessels.append(vessel)

    def handle_context_selection(self, selected_option):
        """This is now the 'Load Scenario' function."""
        sw = self.game_manager.graphics_manager.width
        sh = self.game_manager.graphics_manager.height

        # --- Clear history for new scenario ---
        print(f"--- Clearing vessels and history for new scenario: {selected_option} ---")
        self.vessels.clear()
        self.history_vessel_data.clear()
        self.history_responses.clear()
        self._sim_time = 0.0
        self.llm_call_count = 0

        if selected_option == "Head-On Scenario":
            self.current_scenario = "head_on";
            head_on_scenario(self, sw, sh);
            self.movement_active = True
        elif selected_option == "Cross Over Scenario":
            self.current_scenario = "cross_over";
            cross_over_scenario(self, sw, sh);
            self.movement_active = True
        elif selected_option == "Over Taking Scenario":
            self.current_scenario = "over_taking";
            over_taking_scenario(self, sw, sh);
            self.movement_active = True
        elif selected_option == "Multi vessel Scenario":
            self.current_scenario = "multi_vessel";
            multi_vessel_scenario(self, sw, sh);
            self.movement_active = True
        elif selected_option == "Multi vessel Scenario2":
            self.current_scenario = "multi_vessel_2";
            multi_vessel_scenario_2(self, sw, sh);
            self.movement_active = True
        elif selected_option == "Traffic Separation Scenario":
            self.current_scenario = "tss";
            traffic_separation_scenario(self, sw, sh);
            self.movement_active = True
        else:
            print(f"Warning: Unknown scenario '{selected_option}' selected.")

    def predict_collision(self, v1, v2, time_horizon_sec=60.0, min_dist_km=0.1):
        # ... (This logic is identical to your other managers) ...
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

    def log_llm_interaction(self, to_query_vessels, prompt, response_json):
        # ... (This logic is identical) ...
        vessel_details = [];
        for v in to_query_vessels:
            color_name = self.color_map.get(v.color, "Unknown");
            vessel_details.append(f"{color_name} (ID: {id(v)})")
        involved_str = ", ".join(vessel_details)
        log_entry = {'llm_call_id': self.llm_call_count, 'simulation_time_s': f"{self._sim_time:.2f}",
                     'involved_vessels': involved_str, 'prompt_data': prompt, 'llm_response_json': response_json}
        try:
            with open(self.llm_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.llm_log_fieldnames);
                writer.writerow(log_entry)
            print(f"--- LLM interaction {self.llm_call_count} logged to {self.llm_log_path} ---")
        except Exception as e:
            print(f"ERROR: Could not write to LLM log file: {e}")

    # --- Note: update_vessels is NOT async in RAG mode ---
    def update_vessels(self, dt):
        if not self.movement_active and not self.goal_queue: return
        self._sim_time += dt
        if dt <= 0: return

        if self.goal_queue:
            for entry in self.goal_queue:
                if isinstance(entry, tuple) and len(entry) == 2:
                    vessel, goal = entry
                else:
                    continue
                if isinstance(vessel, int): vessel = next((v for v in self.vessels if id(v) == vessel), None)
                if not vessel: continue
                vessel.goal = goal
            self.goal_queue.clear()
        if not self.movement_active: return

        # 1. Execute & Update State (based on last frame's decision)
        any_move = False
        for v in self.vessels:
            is_tss = (self.current_scenario == 'tss' and isinstance(v, Francisco))
            v.update_heading_and_speed(dt, is_tss)
            v.update_position(dt)
            if v.goal is None and not v.in_maneuver and v.speed < 0.1: v.speed = 0
            if v.goal or v.speed > 0.1: any_move = True
        self.movement_active = any_move

        # 2. Calculate New Velocities (for this frame)
        for v in self.vessels: v.calculate_desired_velocity()

        # 3. Predict Conflicts (using current velocities)
        to_query = []
        MOVING_THRESHOLD_KMPH = 0.1
        for v in self.vessels:
            speed_kmh = (v.speed / self.pixels_per_km) * 3600
            if speed_kmh <= MOVING_THRESHOLD_KMPH: v.in_maneuver = False; continue
            conflict_found = False
            for other_vessel in self.vessels:
                if v is other_vessel: continue
                other_speed_kmh = (other_vessel.speed / self.pixels_per_km) * 3600
                if other_speed_kmh <= MOVING_THRESHOLD_KMPH: continue
                current_dist_km = math.hypot(v.x - other_vessel.x, v.y - other_vessel.y) / self.pixels_per_km
                if current_dist_km > self.radar_range_km: continue
                if self.predict_collision(v, other_vessel): conflict_found = True; break

            # Grace Period Logic
            if conflict_found:
                v.in_maneuver = True
                # RAG mode controls ALL vessels, so no 'isinstance' check
                if self._sim_time - self.llm_cooldown.get(id(v), 0.0) >= self.llm_cooldown_duration:
                    to_query.append(v)
            else:
                last_query_time = self.llm_cooldown.get(id(v), -999)
                grace_period = self.llm_cooldown_duration + 5.0
                if self._sim_time - last_query_time > grace_period:
                    v.in_maneuver = False
                    v.current_maneuver = None

        # 4. Decide (Get decisions for next frame)
        if to_query:
            for v in to_query: self.llm_cooldown[id(v)] = self._sim_time

            context_vessels = set(to_query)
            for vessel_in_query in to_query:
                for other_vessel in self.vessels:
                    if vessel_in_query is not other_vessel and self.predict_collision(vessel_in_query, other_vessel):
                        context_vessels.add(other_vessel)

            current_vessel_states = []
            for v_ctx in context_vessels:
                current_vessel_states.append({
                    "id": id(v_ctx), "pos": (f"{v_ctx.x:.1f}", f"{v_ctx.y:.1f}"),
                    "heading_deg": f"{math.degrees(v_ctx.heading):.1f}",
                    "speed_kmh": f"{(v_ctx.speed / self.pixels_per_km * 3600):.1f}"
                })

            prompt_history_data = list(self.history_vessel_data)
            response_history_data = list(self.history_responses)

            # Note: RAG mode does not have a "TSS" prompt, it uses the standard prompt.
            print(f"===== Generating standard prompt with history (last {len(prompt_history_data)} steps)... =====")
            text_prompt_for_llm = generate_vessel_prompt(
                to_query, list(context_vessels), self.pixels_per_km,
                previous_vessel_data_list=prompt_history_data,
                previous_responses=response_history_data
            )

            print("===== Calling Text RAG LLM... =====")
            # --- Call the RAG system (synchronous) ---
            raw = self.rag_system.get_llm_decision(text_prompt_for_llm)
            self._last_response = raw
            print("Raw LLM response:", raw)
            self.llm_call_count += 1
            # No screenshot in RAG mode

            if raw:
                self.log_llm_interaction(to_query, text_prompt_for_llm, raw)
                parsed = parse_llm_response_for_all(raw)

                if parsed:
                    self.history_vessel_data.append(current_vessel_states)
                    self.history_responses.append(parsed)

                    print("\n--- FINAL PARSED ACTIONS ---")
                    for entry in parsed:
                        vessel_to_update = next((v for v in self.vessels if id(v) == entry['id']), None)
                        if vessel_to_update:
                            vessel_to_update.last_situation = entry.get('situation', '')
                            vessel_to_update.last_role = entry.get('role', '')
                            maneuver_enum = entry['maneuver']
                            vessel_to_update.set_maneuver(maneuver_enum)
                            color_name = self.color_map.get(vessel_to_update.color, "Unknown")
                            print(
                                f"  > {color_name} Vessel (ID: {entry['id']}): Assigned Maneuver -> {maneuver_enum.name}")
                    print("--------------------------\n")