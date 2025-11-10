import math
from typing import List, Optional, Set, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.utils import seeding


class BaseBoat:
    """Rigid body with position (x,y), speed, and heading (radians)."""

    def __init__(self, x: float, y: float, speed: float, heading: float):
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading

    def step(self, dt: float):
        """Integrate constant-velocity motion for time step dt."""
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt


class ControlledBoat(BaseBoat):
    """Agent-controlled boat with turn rate and acceleration."""

    def __init__(
        self,
        x: float,
        y: float,
        speed: float,
        max_speed: int,
        heading: float,
        turn_rate: float,
        acceleration: float
    ):
        super().__init__(x, y, speed, heading)
        self.max_speed = max_speed
        self.turn_rate = turn_rate
        self.acceleration = acceleration

    def act(self, action: int, dt: float):
        """
        Action (steer, throttle):
          steer:    0=straight, 1=right, 2=left
          throttle: 0=coast,    1=accelerate, 2=decelerate
        """
        steer, throttle = action

        # Turn only while moving (preserved behavior).
        if steer == 1 and self.speed > 0:
            self.heading -= self.turn_rate * dt
        elif steer == 2 and self.speed > 0:
            self.heading += self.turn_rate * dt

        # Wrap heading to [-pi, pi].
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # Throttle.
        if throttle == 1:
            self.speed = min(self.speed + self.acceleration * dt, self.max_speed)
        elif throttle == 2:
            self.speed = max(self.speed - self.acceleration * dt, 0.0)

        self.step(dt)


class TrafficBoat(BaseBoat):
    """Traffic boat moving in a fixed lane direction. Can be 'deleted' for the episode."""

    def __init__(self, x: float, y: float, speed: float, direction: str):
        heading = 0.0 if direction == "east" else math.pi
        super().__init__(x, y, speed, heading)
        self.direction = direction          # "east" or "west"
        self.active = True                  # currently in play
        self.respawn_timer = 0.0            # if inactive, steps until respawn
        self.deleted = False                # if True, never respawn and ignored everywhere


class ChannelCrossingEnv(Env):
    """
    Boat crosses a channel without colliding with lane traffic.

    HUMAN mode controls:
      - P: pause/resume
      - While paused:
          Left-click             : select boat
          Shift + Left-click     : add boat on a lane at mouse X (direction by E/W)
          Right-click            : clear selection
          Ctrl+A                 : select all active traffic
          E / W                  : choose direction for adding (east/west)
          D/Delete/Backspace     : delete selected (no respawn this episode)
          Esc                    : clear selection
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    BOAT_VERTS = np.array([
        [ +3.0,  0.0 ],
        [ -3.0, -2.0 ],
        [ -3.0, +2.0 ]
    ], dtype=float)

    BOAT_COLLISION_RADIUS = max(
        math.hypot(vx, vy) for vx, vy in BOAT_VERTS
    )

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        traffic_low: float = 30.0,
        traffic_high: float = 70.0,
        channel_padding: float = 10.0,
        n_traffic: int = 6,
        traffic_speed: float = 8.0 * 0.514,
        agent_speed: float   = 40.0 * 0.514,
        turn_rate = math.pi / 3,
        agent_acceleration: int = 2,
        max_steps: int = 500,
        render_mode: str = None,
        window_name: Optional[str] = None
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.traffic_low = traffic_low
        self.traffic_high = traffic_high
        self.channel_padding = channel_padding
        self.channel_low = self.traffic_low - self.channel_padding
        self.channel_high = self.traffic_high + self.channel_padding

        self.goal_x = None
        self.goal_y = None

        self.n_traffic = n_traffic
        self.agent_speed = agent_speed
        self.turn_rate = turn_rate
        self.agent_acceleration = agent_acceleration
        self.traffic_speed = traffic_speed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.window_name = window_name or "Channel Crossing"

        verts = self.BOAT_VERTS
        self.boat_half_length = float(np.max(verts[:, 0]))
        self.boat_half_width  = float(np.max(np.abs(verts[:, 1])))

        self.living_reward = 0.01

        self.screen = None
        self.screen_size = 600
        self.scale = self.screen_size / self.width
        self.clock = None
        self.font = None

        self.paused: bool = False
        self.selected: Set[int] = set()
        self.add_direction: str = "east"
        self.lane_ys: List[float] = []

        self.action_space = spaces.Discrete(3 * 3)
        obs_dim = 3 + 4 * self.n_traffic
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high=np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32
        )

        self.agent = None
        self.traffic: List[TrafficBoat] = []
        self.prev_distance = 0.0
        self.prev_y = 0.0
        self.steps = 0

        self.fps = self.metadata["render_fps"]
        self.dt = 1.0 / float(self.fps)

    # -------- Geometry helpers (halo overlap, picking) --------

    def decode_action(self, i):
        """Inverse of action encoding: 0..8 → (steer, throttle)."""
        steer = i // 3
        throttle = i % 3
        return steer, throttle

    def _ellipse_params(
        self,
        boat: BaseBoat,
        margin: float = 1.2,
        base_offset_ratio: float = 0.2
    ) -> Tuple[float, float, float, float]:
        """Halo ellipse (a,b) centered slightly aft of the bow."""
        a = self.boat_half_length * margin
        b = self.boat_half_width * margin
        off = -base_offset_ratio * a
        cx = boat.x + off * math.cos(boat.heading)
        cy = boat.y + off * math.sin(boat.heading)
        return a, b, cx, cy

    def _draw_halo(self, surf, boat: BaseBoat, color=(255, 0, 0, 80), width: int = 2):
        """Render the boat's oriented safety halo."""
        a, b, cx, cy = self._ellipse_params(boat)
        w_px = int(2 * a * self.scale)
        h_px = int(2 * b * self.scale)
        halo = pygame.Surface((w_px, h_px), flags=pygame.SRCALPHA)
        pygame.draw.ellipse(halo, color, halo.get_rect(), width=width)
        rotated = pygame.transform.rotate(halo, math.degrees(boat.heading))
        px = cx * self.scale
        py = (self.height - cy) * self.scale
        rect = rotated.get_rect(center=(px, py))
        surf.blit(rotated, rect)

    def _check_halo_overlap(self, boat1: BaseBoat, boat2: BaseBoat) -> bool:
        """Approximate collision: overlap of the two safety halos."""
        a1, b1, cx1, cy1 = self._ellipse_params(boat1)
        a2, b2, cx2, cy2 = self._ellipse_params(boat2)
        a_comb = a1 + a2
        b_comb = b1 + b2
        dx = cx2 - cx1
        dy = cy2 - cy1
        ch = math.cos(boat1.heading)
        sh = math.sin(boat1.heading)
        x_prime =  ch * dx + sh * dy
        y_prime = -sh * dx + ch * dy
        return (x_prime / a_comb) ** 2 + (y_prime / b_comb) ** 2 <= 1.0

    def _point_in_halo(self, boat: BaseBoat, wx: float, wy: float) -> bool:
        """Test if world point (wx,wy) lies inside a boat's halo."""
        a, b, cx, cy = self._ellipse_params(boat)
        dx = wx - cx
        dy = wy - cy
        ch = math.cos(boat.heading)
        sh = math.sin(boat.heading)
        x_prime =  ch * dx + sh * dy
        y_prime = -sh * dx + ch * dy
        return (x_prime / a) ** 2 + (y_prime / b) ** 2 <= 1.0

    def _pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert screen pixel to world coordinates."""
        wx = px / self.scale
        wy = self.height - (py / self.scale)
        return wx, wy

    def _pick_traffic_index_at_pixel(self, pos: Tuple[int, int]) -> Optional[int]:
        """Return index of first active, non-deleted traffic boat under cursor."""
        mx, my = pos
        wx, wy = self._pixel_to_world(mx, my)
        for i, tb in enumerate(self.traffic):
            if tb.deleted or not tb.active:
                continue
            if self._point_in_halo(tb, wx, wy):
                return i
        return None

    # -------- Lane helpers + "add boat" plumbing --------

    def _lane_index_from_world_y(self, wy: float, tol_px: int = 12) -> Optional[int]:
        """Nearest lane index to world Y within pixel tolerance."""
        if not self.lane_ys:
            return None
        tol_world = tol_px / self.scale
        best_idx = None
        best_dy = float("inf")
        for i, ly in enumerate(self.lane_ys):
            dy = abs(wy - ly)
            if dy < best_dy:
                best_dy = dy
                best_idx = i
        return best_idx if best_dy <= tol_world else None

    def _add_boat_at(self, wx: float, lane_idx: int, direction: str) -> bool:
        """
        Revive a deleted/inactive traffic slot at X=wx on the lane (keeps obs size fixed).
        Returns True if a slot was used, False if none available.
        """
        if lane_idx is None or not (0 <= lane_idx < len(self.lane_ys)):
            return False
        wy = self.lane_ys[lane_idx]
        heading = 0.0 if direction == "east" else math.pi

        # Prefer a deleted slot; otherwise reuse an inactive one awaiting respawn.
        candidate = None
        for tb in self.traffic:
            if tb.deleted:
                candidate = tb
                break
        if candidate is None:
            for tb in self.traffic:
                if not tb.active:
                    candidate = tb
                    break
        if candidate is None:
            return False

        candidate.x = float(wx)
        candidate.y = float(wy)
        candidate.speed = self.traffic_speed
        candidate.direction = direction
        candidate.heading = heading
        candidate.deleted = False
        candidate.active = True
        candidate.respawn_timer = 0.0
        return True

    # -------- Interactive handlers (pause/select/delete/add) --------

    def _delete_selected(self):
        """Permanently delete selected traffic boats for this episode."""
        for idx in list(self.selected):
            if 0 <= idx < len(self.traffic):
                tb = self.traffic[idx]
                tb.deleted = True
                tb.active = False
                tb.respawn_timer = 10**12
                tb.x, tb.y = -1000.0, -1000.0
                tb.speed = 0.0
        self.selected.clear()

    # === NEW: programmatic deletion helpers (same effect as manual pause+delete) ===
    def delete_traffic_indices(self, indices, permanent: bool = True):
        """Programmatically delete specific traffic boats by index (0..n_traffic-1)."""
        for idx in indices:
            if 0 <= idx < len(self.traffic):
                tb = self.traffic[idx]
                tb.deleted = True
                tb.active = False
                tb.respawn_timer = 10**12 if permanent else 0.0
                tb.x, tb.y = -1000.0, -1000.0
                tb.speed = 0.0

    def delete_traffic_mask(self, mask: int, permanent: bool = True):
        """
        Delete using a 6-bit mask (bit i==1 → delete traffic[i]).
        Works with any n_traffic; higher bits are ignored.
        """
        n = min(self.n_traffic, len(self.traffic))
        to_delete = [i for i in range(n) if (mask >> i) & 1]
        self.delete_traffic_indices(to_delete, permanent=permanent)

    def traffic_index_info(self):
        """(Optional) Helps you see which lane/index each boat is."""
        return [
            {
                "index": i,
                "y": float(tb.y),
                "direction": tb.direction,
                "active": bool(tb.active),
                "deleted": bool(tb.deleted),
            }
            for i, tb in enumerate(self.traffic)
        ]

    def _handle_events(self):
        """Keyboard/mouse input handling for HUMAN mode."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_p:
                    self.paused = not self.paused
                elif self.paused and ev.key in (pygame.K_d, pygame.K_DELETE, pygame.K_BACKSPACE):
                    self._delete_selected()
                elif self.paused and ev.key == pygame.K_ESCAPE:
                    self.selected.clear()
                elif self.paused and ev.key == pygame.K_a and (ev.mod & pygame.KMOD_CTRL):
                    self.selected = {i for i, tb in enumerate(self.traffic) if tb.active and not tb.deleted}
                elif self.paused and ev.key == pygame.K_e:
                    self.add_direction = "east"
                elif self.paused and ev.key == pygame.K_w:
                    self.add_direction = "west"

            elif self.paused and ev.type == pygame.MOUSEBUTTONDOWN:
                mods = pygame.key.get_mods()  # <-- FIX: mouse events don't have ev.mod
                # Shift + Left-click → add boat on nearest lane at mouse X
                if ev.button == 1 and (mods & pygame.KMOD_SHIFT):
                    mx, my = ev.pos
                    wx, wy = self._pixel_to_world(mx, my)
                    lane_idx = self._lane_index_from_world_y(wy)
                    self._add_boat_at(wx, lane_idx, self.add_direction)
                elif ev.button == 1:
                    # Normal select/unselect toggle
                    idx = self._pick_traffic_index_at_pixel(ev.pos)
                    if idx is not None:
                        if idx in self.selected:
                            self.selected.remove(idx)
                        else:
                            self.selected.add(idx)
                elif ev.button == 3:  # clear
                    self.selected.clear()

    # ---------------- Gym API ----------------

    def reset(self, seed=None, options=None):
        """Spawn agent below the channel and traffic boats in lanes."""
        super().reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)
        self.steps = 0
        self.paused = False
        self.selected.clear()
        self.add_direction = "east"

        self.goal_x = self.np_random.uniform(0, self.width)
        self.goal_y = self.channel_high + self.channel_padding

        agent_x = self.np_random.uniform(0.0, self.width)
        agent_y = self.channel_low - self.channel_padding
        agent_heading = self.np_random.uniform(-math.pi, math.pi)
        self.agent = ControlledBoat(
            x=agent_x,
            y=agent_y,
            speed=self.agent_speed,
            max_speed=self.agent_speed,
            heading=agent_heading,
            turn_rate=self.turn_rate,
            acceleration=self.agent_acceleration
        )

        self.lane_ys = list(np.linspace(self.traffic_low, self.traffic_high, self.n_traffic))
        traffic_mid = self.n_traffic // 2
        self.traffic = []
        for n, traffic_y in enumerate(self.lane_ys):
            direction = "west" if n < traffic_mid else "east"
            traffic_x = self.np_random.uniform(0.0, self.width)
            tb = TrafficBoat(traffic_x, traffic_y, self.traffic_speed, direction)
            tb.active = True
            tb.respawn_timer = 0
            tb.deleted = False
            self.traffic.append(tb)

        self.prev_distance = math.hypot(self.goal_x - self.agent.x, self.goal_y - self.agent.y)
        self.prev_y = self.agent.y

        return self._get_obs(), {}

    def step(self, action: int):
        """Advance simulation one control step, compute reward, and check terminals."""
        self.steps += 1

        steer, throttle = self.decode_action(action)
        self.agent.act((steer, throttle), self.dt)

        min_delay, max_delay = 30, 90
        margin = 3.0
        for tb in self.traffic:
            if tb.deleted:
                continue
            if tb.active:
                tb.step(self.dt)
                if tb.x < -margin or tb.x > self.width + margin:
                    tb.active = False
                    tb.respawn_timer = self.np_random.integers(min_delay, max_delay)
            else:
                tb.respawn_timer -= 1
                if tb.respawn_timer <= 0:
                    tb.active = True
                    tb.x = 0.0 if tb.direction == "east" else self.width

        distance = math.hypot(self.goal_x - self.agent.x, self.goal_y - self.agent.y)
        step_reward_distance = -1 if distance >= self.prev_distance else 1

        if self.agent.y < self.channel_high:
            _dy = self.agent.y - self.prev_y
        self.prev_distance = distance
        self.prev_y = self.agent.y

        reward = step_reward_distance - self.living_reward

        done = False
        for tb in self.traffic:
            if tb.deleted or not tb.active:
                continue
            if self._check_halo_overlap(self.agent, tb):
                reward -= 50.0
                done = True
                break

        if not done and distance < 5.0:
            reward += 50.0
            done = True

        if not done and (self.agent.x < 0 or self.agent.x > self.width or self.agent.y < 0 or self.agent.y > self.height):
            done = True
        if not done and self.steps >= self.max_steps:
            done = True

        return self._get_obs(), float(reward), done, False, {}

    def _get_obs(self) -> np.ndarray:
        """
        Observation:
          [agent_x, agent_y, agent_heading,
           per-traffic: (rx, ry, rvx, rvy) * n_traffic]
        Deleted boats contribute far-away, stationary values to keep shape fixed.
        """
        obs = [self.agent.x, self.agent.y, self.agent.heading]
        for tb in self.traffic:
            if tb.deleted:
                rx, ry, rvx, rvy = 1e3, 1e3, 0.0, 0.0
            else:
                rx = tb.x - self.agent.x
                ry = tb.y - self.agent.y
                rvx = tb.speed * math.cos(tb.heading) - self.agent.speed * math.cos(self.agent.heading)
                rvy = tb.speed * math.sin(tb.heading) - self.agent.speed * math.sin(self.agent.heading)
            obs += [rx, ry, rvx, rvy]
        return np.array(obs, dtype=np.float32)

    def _compute_tcpa_dcpa(self, a: BaseBoat, b: BaseBoat) -> Tuple[float, float]:
        """Time/Distance at Closest Point of Approach between two moving boats."""
        rx, ry = b.x - a.x, b.y - a.y
        rvx = b.speed * math.cos(b.heading) - a.speed * math.cos(a.heading)
        rvy = b.speed * math.sin(b.heading) - a.speed * math.sin(a.heading)
        rv2 = rvx * rvx + rvy * rvy
        if rv2 < 1e-8:
            return 0.0, math.hypot(rx, ry)
        tcpa = max(0.0, - (rx * rvx + ry * rvy) / rv2)
        cx, cy = rx + rvx * tcpa, ry + rvy * tcpa
        return tcpa, math.hypot(cx, cy)

    def render(self):
        """Draw the scene (HUMAN) or return an RGB array (rgb_array)."""
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption(self.window_name)
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 22)
            surface = self.screen
            self._handle_events()
        elif self.render_mode == "rgb_array":
            surface = pygame.Surface((self.screen_size, self.screen_size))
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

        surface.fill((255, 255, 255))
        line_y1 = int((self.height - self.channel_low) * self.scale)
        line_y2 = int((self.height - self.channel_high) * self.scale)
        pygame.draw.line(surface, (200, 0, 0), (0, line_y1), (self.screen_size, line_y1), 3)
        pygame.draw.line(surface, (200, 0, 0), (0, line_y2), (self.screen_size, line_y2), 3)

        goal_px = int(self.goal_x * self.scale)
        goal_py = int((self.height - self.goal_y) * self.scale)
        pygame.draw.circle(surface, (0, 0, 0), (goal_px, goal_py), 7)

        for i, boat in enumerate([*self.traffic, self.agent]):
            if isinstance(boat, TrafficBoat):
                if boat.deleted or not boat.active:
                    continue
                if i in self.selected:
                    self._draw_halo(surface, boat, color=(255, 220, 0, 140), width=4)
                else:
                    self._draw_halo(surface, boat)
            else:
                self._draw_halo(surface, boat)

        for i, tb in enumerate(self.traffic):
            if tb.deleted or not tb.active:
                continue
            self._draw_boat(surface, tb.x, tb.y, tb.heading, (0, 0, 200))
        self._draw_boat(surface, self.agent.x, self.agent.y, self.agent.heading, (0, 200, 0))

        if self.paused and self.render_mode == "human":
            overlay = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
            overlay.fill((30, 30, 30, 120))

            guide_color = (0, 180, 255, 170)
            for ly in self.lane_ys:
                py = int((self.height - ly) * self.scale)
                pygame.draw.line(overlay, guide_color, (0, py), (self.screen_size, py), 2)

            if self.font:
                big = pygame.font.Font(None, 48)
                t1 = big.render("PAUSED", True, (255, 255, 255))
                surface.blit(overlay, (0, 0))
                surface.blit(t1, t1.get_rect(center=(self.screen_size // 2, self.screen_size // 2 - 40)))

                hint_lines = [
                    "Click: select | D/Delete: remove | Right-click: clear | P: resume",
                    "Shift+Click lane: add boat at X | E/W: set direction | Ctrl+A: select all",
                    f"Add direction: {self.add_direction.upper()}",
                ]
                y = self.screen_size // 2 + 5
                for line in hint_lines:
                    txt = self.font.render(line, True, (230, 230, 230))
                    rect = txt.get_rect(center=(self.screen_size // 2, y))
                    surface.blit(txt, rect)
                    y += 24
            else:
                surface.blit(overlay, (0, 0))

        arr = pygame.surfarray.array3d(surface)  # (W,H,3)
        frame = np.transpose(arr, (1, 0, 2))     # → (H,W,3)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        return frame

    def _draw_boat(self, surf, bx, by, heading, color):
        """Render a triangular boat at (bx,by) with heading."""
        pts = []
        for vx, vy in self.BOAT_VERTS:
            wx = bx + vx * math.cos(heading) - vy * math.sin(heading)
            wy = by + vx * math.sin(heading) + vy * math.cos(heading)
            px = int(wx * self.scale)
            py = int((self.height - wy) * self.scale)
            pts.append((px, py))
        pygame.draw.polygon(surf, color, pts)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
