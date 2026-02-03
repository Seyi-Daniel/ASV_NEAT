import pygame
from typing import Optional, List

# UI Configuration ---
pygame.font.init()
# Slightly smaller fonts for a compact look
FONT_UI_TITLE = pygame.font.Font(None, 20)
FONT_UI_OPTION = pygame.font.Font(None, 20)
FONT_UI_BUTTON = pygame.font.Font(None, 20)

COLOR_BG = (40, 44, 52)  # Slightly darker, modern background
COLOR_TEXT = (220, 220, 220)
COLOR_BORDER = (60, 64, 72)
COLOR_BUTTON = (70, 74, 82)
COLOR_BUTTON_HOVER = (90, 94, 102)
COLOR_BUTTON_ACTIVE = (60, 100, 160)  # Active/Selected
COLOR_BUTTON_ACTION = (70, 130, 70)  # Green for Load
COLOR_BUTTON_ACTION_HOVER = (90, 150, 90)
COLOR_DROPDOWN_BG = (50, 54, 62)
COLOR_DROPDOWN_HOVER = (80, 84, 92)
COLOR_PAUSE = (160, 70, 70)  # Red
COLOR_ARROW = (180, 180, 180)  # Color for the dropdown arrow


class Button:
    """A simple clickable UI button (used for Load/Pause)."""

    def __init__(self, x, y, w, h, text, value):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.value = value
        self.is_hovered = False
        self.is_toggle = False
        self.text_surf = FONT_UI_BUTTON.render(text, True, COLOR_TEXT)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def update_text(self, new_text):
        self.text = new_text
        self.text_surf = FONT_UI_BUTTON.render(new_text, True, COLOR_TEXT)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.is_hovered:
                return True
        return False

    def draw(self, screen):
        color = COLOR_BUTTON
        if self.is_toggle and self.value == "paused":
            color = COLOR_BUTTON  # Play color (default)
        elif self.is_toggle and self.value == "running":
            color = COLOR_PAUSE  # Pause color (red)
        elif self.value == "load":
            color = COLOR_BUTTON_ACTION

        if self.is_hovered:
            # Simple brighten effect
            color = tuple(min(c + 20, 255) for c in color)

        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER, self.rect, 1, border_radius=4)
        self.text_rect.center = self.rect.center
        screen.blit(self.text_surf, self.text_rect)


class Dropdown:
    """A clickable dropdown menu with a visual arrow."""

    def __init__(self, x, y, w, h, title, options: List[tuple[str, str]]):
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.options = options
        self.selected_index = 0
        self.is_open = False
        self.is_hovered = False
        self.hovered_option_index = -1

        # Calculate option rects
        self.option_height = 25  # Slightly more compact
        self.dropdown_rect = pygame.Rect(x, y + h, w, len(options) * self.option_height)

    def handle_event(self, event) -> bool:
        """Returns True if selection changed."""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            self.hovered_option_index = -1
            if self.is_open:
                if self.dropdown_rect.collidepoint(event.pos):
                    local_y = event.pos[1] - self.dropdown_rect.y
                    self.hovered_option_index = local_y // self.option_height
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.is_hovered:
                    self.is_open = not self.is_open
                elif self.is_open and self.dropdown_rect.collidepoint(event.pos):
                    if 0 <= self.hovered_option_index < len(self.options):
                        self.selected_index = self.hovered_option_index
                        self.is_open = False
                        return True
                else:
                    self.is_open = False
        return False

    def get_selected_value(self):
        return self.options[self.selected_index][1]

    def draw_arrow(self, screen, color):
        """Draws a small downward triangle on the right side."""
        center_x = self.rect.right - 15
        center_y = self.rect.centery
        # Triangle points: (left, top), (right, top), (center, bottom)
        points = [
            (center_x - 4, center_y - 2),
            (center_x + 4, center_y - 2),
            (center_x, center_y + 3)
        ]
        pygame.draw.polygon(screen, color, points)

    def draw(self, screen):
        # Draw Main Box
        color = COLOR_DROPDOWN_HOVER if self.is_hovered or self.is_open else COLOR_BUTTON
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER, self.rect, 1, border_radius=4)

        # Draw Arrow
        self.draw_arrow(screen, COLOR_ARROW)

        # Draw Label (Title + Selected Option)
        # Format: "Mode: Vision" - Compact display
        full_text = f"{self.title}: {self.options[self.selected_index][0]}"
        text_surf = FONT_UI_OPTION.render(full_text, True, COLOR_TEXT)
        # Left align text with padding, make sure it doesn't overlap arrow
        text_rect = text_surf.get_rect(midleft=(self.rect.x + 8, self.rect.centery))

        # Optional: Clip text if it's too long (simple approach)
        if text_rect.width > self.rect.width - 25:
            # If text is too long, we could truncate it, but for now let's trust layout
            pass

        screen.blit(text_surf, text_rect)

        # Draw Dropdown List (if open)
        if self.is_open:
            pygame.draw.rect(screen, COLOR_DROPDOWN_BG, self.dropdown_rect)
            pygame.draw.rect(screen, COLOR_BORDER, self.dropdown_rect, 1)

            for i, (label, value) in enumerate(self.options):
                opt_rect = pygame.Rect(self.rect.x, self.dropdown_rect.y + (i * self.option_height), self.rect.width,
                                       self.option_height)

                if i == self.hovered_option_index:
                    pygame.draw.rect(screen, COLOR_BUTTON_ACTIVE, opt_rect)

                opt_surf = FONT_UI_OPTION.render(label, True, COLOR_TEXT)
                opt_text_rect = opt_surf.get_rect(midleft=(opt_rect.x + 8, opt_rect.centery))
                screen.blit(opt_surf, opt_text_rect)


class UIManager:
    def __init__(self, screen_width):
        self.dropdowns = {}
        self.action_buttons = {}

        # Compact Nav Bar (Height 60)
        self.nav_bar_rect = pygame.Rect(0, 0, screen_width, 60)

        # --- LAYOUT CONFIGURATION ---
        y_pos = 15
        height = 30
        gap = 15

        # Positioning logic
        current_x = 10

        # 1. Mode Dropdown
        modes = [("Standard", "standard"), ("RAG", "rag"), ("Vision", "vision")]
        self.dropdowns['mode'] = Dropdown(current_x, y_pos, 140, height, "Mode", modes)
        current_x += 135 + gap

        # 2. LLM Dropdown
        llms = [("GPT-4o", "openai"), ("Claude 3", "claude"), ("DeepSeek", "deepseek")]
        self.dropdowns['llm'] = Dropdown(current_x, y_pos, 130, height, "LLM", llms)
        current_x += 130 + gap

        # 3. Prompt Type Dropdown
        prompts = [("Minimal", "minimal"), ("Moderate", "moderate"), ("Detailed", "detailed"), ("Natural", "natural"),
                   ("TSS", "tss")]
        self.dropdowns['prompt'] = Dropdown(current_x, y_pos, 150, height, "Prompt", prompts)
        current_x += 150 + gap

        # 4. Scenario Dropdown
        scenarios = [
            ("Head-On", "Head-On Scenario"),
            ("Crossing", "Cross Over Scenario"),
            ("Overtaking", "Over Taking Scenario"),
            ("Multi 1", "Multi vessel Scenario"),
            ("Multi 2", "Multi vessel Scenario2"),
            ("TSS", "Traffic Separation Scenario")
        ]
        self.dropdowns['scenario'] = Dropdown(current_x, y_pos, 160, height, "Scene", scenarios)

        # Buttons (Right Aligned)
        btn_width = 70
        pause_x = screen_width - btn_width - 10
        load_x = pause_x - btn_width - 10

        self.action_buttons['load'] = Button(load_x, y_pos, btn_width, height, "LOAD", "load")
        self.action_buttons['load'].is_action = True

        self.action_buttons['pause'] = Button(pause_x, y_pos, btn_width, height, "PAUSE", "running")
        self.action_buttons['pause'].is_toggle = True

    def handle_events(self, events):
        actions = {'load': False, 'pause': False}
        for event in events:
            # Dropdown logic: if one captures the event, don't let others handle it (prevents click-through)
            dropdown_captured = False
            # Process in reverse order so if multiple are open/overlapping (unlikely in horizontal bar), top one wins
            # For horizontal layout it doesn't matter much, but good practice.
            for key in reversed(list(self.dropdowns.keys())):
                if self.dropdowns[key].handle_event(event):
                    dropdown_captured = True
                    break  # Stop processing other UI if a dropdown was clicked

            if not dropdown_captured:
                if self.action_buttons['load'].handle_event(event): actions['load'] = True
                if self.action_buttons['pause'].handle_event(event):
                    btn = self.action_buttons['pause']
                    if btn.value == "running":
                        btn.value = "paused";
                        btn.update_text("RESUME")
                    else:
                        btn.value = "running";
                        btn.update_text("PAUSE")
                    actions['pause'] = True

        return actions

    def draw(self, screen):
        # Draw Background
        pygame.draw.rect(screen, COLOR_BG, self.nav_bar_rect)
        pygame.draw.line(screen, COLOR_BORDER, (0, 60), (screen.get_width(), 60), 1)

        # Draw Buttons
        self.action_buttons['load'].draw(screen)
        self.action_buttons['pause'].draw(screen)

        # Draw Dropdowns
        # Important: Draw them in order so their menus render on top of neighbors if needed
        for dd in self.dropdowns.values(): dd.draw(screen)

    def get_value(self, key: str) -> Optional[str]:
        if key in self.dropdowns: return self.dropdowns[key].get_selected_value()
        return None