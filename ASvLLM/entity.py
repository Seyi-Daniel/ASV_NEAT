import pygame
import math
from response_parser import Maneuver

# Colors and font init...
WHITE = (255, 255, 255);
BLACK = (0, 0, 0);
RED = (255, 0, 0)
GREEN = (0, 255, 0);
BLUE = (0, 0, 255);
GRAY = (200, 200, 200)
pygame.font.init();
font = pygame.font.Font(None, 24)


def kmph_to_pixels_per_sec(kmph, pixels_per_km):
    if pixels_per_km <= 0: raise ValueError("pixels_per_km must be positive")
    return (kmph / 3600) * pixels_per_km


# --- ContextMenu class is REMOVED ---

class Vessel:
    def __init__(self, x, y, pixels_per_km, size=25, color=RED):
        self.x, self.y = float(x), float(y)
        self.size, self.color = size, color
        self.speed, self.acceleration = 0.0, 2.028
        self.pixels_per_km = pixels_per_km
        self.max_speed_kmph = 67
        self.max_speed = kmph_to_pixels_per_sec(self.max_speed_kmph, pixels_per_km)
        self.slow_down_distance = 5.0
        self.goal = None
        self.selected = False
        self.heading = 0.0
        self.rect = pygame.Rect(self.x - self.size / 2, self.y - self.size / 2, self.size, self.size)
        self.desired_velocity = [0.0, 0.0]
        self.turn_rate = math.radians(10)
        self.maneuver_turn_rate = math.radians(20)
        self.current_maneuver = None
        self.in_maneuver = False
        self.maneuver_target_heading = None
        self.maneuver_turn_degrees = 30.0
        self.tss_target_heading = 0.0

    @staticmethod
    def rotate_point(point, angle, origin):
        ox, oy = origin;
        px, py = point
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def draw(self, screen):
        tip = (self.x, self.y - self.size / 2)
        left = (self.x - self.size / 3, self.y + self.size / 2)
        right = (self.x + self.size / 3, self.y + self.size / 2)
        pts = [Vessel.rotate_point(p, self.heading, (self.x, self.y)) for p in [tip, left, right]]
        pygame.draw.polygon(screen, self.color, pts)
        if self.selected: pygame.draw.polygon(screen, GREEN, pts, 3)
        if self.goal: pygame.draw.circle(screen, BLUE, self.goal, 5)

    def set_selected(self, val):
        self.selected = val

    def contains_point(self, pos):
        return self.rect.collidepoint(pos)

    def _turn_towards_heading(self, target_heading, turn_rate, dt):
        diff = (target_heading - self.heading + math.pi) % (2 * math.pi) - math.pi
        max_turn = turn_rate * dt
        if abs(diff) > max_turn:
            self.heading += max_turn if diff > 0 else -max_turn
        else:
            self.heading = target_heading
        self.heading %= (2 * math.pi)

    def set_maneuver(self, maneuver: Maneuver):
        self.current_maneuver = maneuver;
        self.maneuver_target_heading = None
        turn_rad = math.radians(self.maneuver_turn_degrees)
        if maneuver == Maneuver.ALTER_COURSE_STARBOARD:
            self.maneuver_target_heading = (self.heading + turn_rad) % (2 * math.pi)
        elif maneuver == Maneuver.ALTER_COURSE_PORT:
            self.maneuver_target_heading = (self.heading - turn_rad) % (2 * math.pi)
        elif maneuver == Maneuver.PASS_ASTERN:
            self.maneuver_target_heading = (self.heading + turn_rad * 1.5) % (2 * math.pi)

    def update_heading_and_speed(self, dt, is_tss_scenario: bool):
        target_heading = self.heading;
        active_turn_rate = self.turn_rate
        if self.in_maneuver:
            active_turn_rate = self.maneuver_turn_rate
            command = self.current_maneuver
            if command == Maneuver.REDUCE_SPEED:
                self.speed = max(self.speed - self.acceleration * dt, 0)
            elif command == Maneuver.ACCELERATE:
                self.speed = min(self.speed + self.acceleration * dt, self.max_speed)
            elif command == Maneuver.PASS_ASTERN:
                self.speed = max(self.speed - self.acceleration * dt, 0)
                if self.maneuver_target_heading is not None: target_heading = self.maneuver_target_heading
            elif command in (Maneuver.ALTER_COURSE_STARBOARD, Maneuver.ALTER_COURSE_PORT):
                if self.maneuver_target_heading is not None: target_heading = self.maneuver_target_heading
        else:
            if is_tss_scenario:
                target_heading = self.tss_target_heading
            elif self.goal:
                dx = self.goal[0] - self.x;
                dy = self.goal[1] - self.y
                target_heading = (math.atan2(dy, dx) + math.pi / 2) % (2 * math.pi)

        self._turn_towards_heading(target_heading, active_turn_rate, dt)

        if self.maneuver_target_heading is not None:
            turn_diff = abs((self.heading - self.maneuver_target_heading + math.pi) % (2 * math.pi) - math.pi)
            if turn_diff < math.radians(1.0):
                self.maneuver_target_heading = None

        is_maneuvering_speed = self.in_maneuver and self.current_maneuver in (Maneuver.REDUCE_SPEED,
                                                                              Maneuver.PASS_ASTERN, Maneuver.ACCELERATE)
        if not is_maneuvering_speed:
            dist = None
            if self.goal: dx, dy = self.goal[0] - self.x, self.goal[1] - self.y; dist = math.hypot(dx, dy)
            if dist is None or dist > self.slow_down_distance:
                self.speed = min(self.speed + self.acceleration * dt, self.max_speed)
            elif dist is not None:
                self.speed = max(self.speed - self.acceleration * dt, 0)

    def calculate_desired_velocity(self):
        vx = self.speed * math.sin(self.heading)
        vy = -self.speed * math.cos(self.heading)
        self.desired_velocity = [vx, vy]

    def update_position(self, dt):
        self.calculate_desired_velocity()
        self.x += self.desired_velocity[0] * dt
        self.y += self.desired_velocity[1] * dt
        self.rect.topleft = (self.x - self.size / 2, self.y - self.size / 2)
        if self.goal:
            dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
            dist = math.hypot(dx, dy)
            if dist < 15: self.goal = None; self.speed = 0


class USSTucker(Vessel):
    def __init__(self, x, y, pixels_per_km):
        super().__init__(x, y, pixels_per_km, size=30, color=(0, 128, 255))
        self.max_speed_kmph = 37
        self.max_speed = kmph_to_pixels_per_sec(self.max_speed_kmph, pixels_per_km)


class Francisco(Vessel):
    def __init__(self, x, y, pixels_per_km):
        super().__init__(x, y, pixels_per_km, size=25, color=BLUE)
        self.max_speed_kmph = 150  # Keeping your faster speed
        self.acceleration = 3.028
        self.max_speed = kmph_to_pixels_per_sec(self.max_speed_kmph, pixels_per_km)
        self.slow_down_distance = 15.0


class USSGerald(Vessel):
    pass