import pygame

class GraphicsManager:
    def __init__(self, width, height, caption, pixels_per_km=1000):
        self.width = width
        self.height = height
        self.pixels_per_km = pixels_per_km
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption)

    def world_to_screen(self, world_x, world_y):
        """Convert world (km) to screen (pixels), with origin at screen center."""
        # This is currently not used, as your simulation uses top-left 0,0
        # But it's good to keep if you ever want to change coordinate systems
        screen_x = self.width // 2 + world_x * self.pixels_per_km
        screen_y = self.height // 2 - world_y * self.pixels_per_km  # Y flipped
        return screen_x, screen_y

    def clear(self, color):
        self.screen.fill(color)

    def update_display(self):
        pygame.display.flip()