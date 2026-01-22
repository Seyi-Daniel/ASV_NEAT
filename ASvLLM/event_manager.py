import pygame
import sys


class EventManager:
    def __init__(self, game_manager):
        self.game_manager = game_manager

    def handle_events(self, events):
        """Process a list of Pygame events."""
        for event in events:
            if event.type == pygame.QUIT:
                self.game_manager.running = False

            # You can add other global keys here, like pause
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.game_manager.running = False