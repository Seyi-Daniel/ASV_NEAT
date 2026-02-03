import pygame
import os
import asyncio
from graphics_manager import GraphicsManager
from event_manager import EventManager
from entity import WHITE
from ui_manager import UIManager
from entity_manager import EntityManager


class GameManager:
    def __init__(self, width, height):
        self.running = True
        self.paused = False
        self.graphics_manager = GraphicsManager(width, height, "ASV")
        self.event_manager = EventManager(self)

        self.ui_manager = UIManager(width)
        self.entity_manager = None
        self.current_mode = None
        self.current_llm = None
        self.current_prompt_type = None

        # We can remove is_async_mode because now EVERYTHING is async
        # self.is_async_mode = False

    def load_simulation(self):
        """
        Called when the 'LOAD SCENARIO' button is pressed.
        Creates a new EntityManager based on the UI selections.
        """
        # 1. Get configuration from UI
        self.current_mode = self.ui_manager.get_value("mode")
        self.current_llm = self.ui_manager.get_value("llm")
        self.current_prompt_type = self.ui_manager.get_value("prompt")
        selected_scenario = self.ui_manager.get_value("scenario")

        print(f"--- Loading Simulation ---")
        print(f"  Mode: {self.current_mode}")
        print(f"  LLM: {self.current_llm}")
        print(f"  Prompt Type: {self.current_prompt_type}")
        print(f"  Scenario: {selected_scenario}")

        # 2. Reset existing manager
        if self.entity_manager:
            self.entity_manager = None

        try:
            # 3. Create the Unified EntityManager
            self.entity_manager = EntityManager(
                self,
                mode=self.current_mode,
                llm_provider=self.current_llm,
                prompt_type=self.current_prompt_type
            )

            # 5. Load the Scenario
            self.entity_manager.load_scenario(selected_scenario)

            # 6. Reset UI State
            self.paused = False
            if 'pause' in self.ui_manager.action_buttons:
                self.ui_manager.action_buttons['pause'].value = "running"
                self.ui_manager.action_buttons['pause'].update_text("PAUSE")

        except Exception as e:
            print(f"ERROR: Failed to load simulation: {e}")
            import traceback
            traceback.print_exc()
            self.entity_manager = None

    async def run_async(self):
        """The main game loop."""
        clock = pygame.time.Clock()
        while self.running:
            events = pygame.event.get()

            # --- Handle Events ---
            self.event_manager.handle_events(events)
            ui_actions = self.ui_manager.handle_events(events)

            if ui_actions['load']:
                self.load_simulation()

            if ui_actions['pause']:
                self.paused = not self.paused
                print(f"Simulation {'PAUSED' if self.paused else 'RESUMED'}")

            # --- Update Simulation ---
            dt = clock.tick(60) / 1000.0
            dt = min(dt, 1 / 30)  # Cap delta time to prevent jumps on lag

            if not self.paused and self.entity_manager:
                # --- CRITICAL FIX HERE ---
                # We ALWAYS await because entity_manager.update_vessels is now async def
                await self.entity_manager.update_vessels(dt)
                # -------------------------

            # --- Render ---
            self.render()
            await asyncio.sleep(0)  # Yield control to event loop

        # --- Cleanup ---
        if self.entity_manager and hasattr(self.entity_manager, 'llm_log_path'):
            print(f"Log saved to: {self.entity_manager.llm_log_path}")
        pygame.quit()

    def render(self):
        """Draws the simulation and the UI."""
        self.graphics_manager.clear(WHITE)

        # Draw the simulation layer
        if self.entity_manager:
            self.entity_manager.draw(self.graphics_manager.screen)

            # Screenshot logic
            if hasattr(self.entity_manager,
                       'take_screenshot_on_next_render') and self.entity_manager.take_screenshot_on_next_render:
                filename = (
                    f"llm_call_{self.entity_manager.llm_call_count}_time_{self.entity_manager._sim_time:.1f}s.png")
                screenshot_dir = getattr(self.entity_manager, 'screenshot_dir', 'screenshots')
                if not os.path.exists(screenshot_dir):
                    os.makedirs(screenshot_dir)
                filepath = os.path.join(screenshot_dir, filename)
                pygame.image.save(self.graphics_manager.screen, filepath)
                print(f"--- Screenshot saved: {filepath} ---")
                self.entity_manager.take_screenshot_on_next_render = False

        # Draw the UI layer on top
        self.ui_manager.draw(self.graphics_manager.screen)
        self.graphics_manager.update_display()