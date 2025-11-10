#!/usr/bin/env python3
"""
Multi-Boat Sectors Environment (turn-session control, kinematic update)
======================================================================

This environment is *physics-light* on purpose:
- No Nomoto yaw, no rudder slew, no drag models, no leeway, no prop wash.
- Heading changes only via a *turn session*: a latched command to rotate by
  a fixed angle (e.g., ±15°) at a fixed yaw rate (e.g., 45°/s).
- Throttle is simple kinematics: accelerate/decelerate linearly (m/s^2),
  clamp speed to [min_speed, max_speed]; no passive drag.

Goal & Observations
-------------------
- Each boat spawns at random (x,y,heading), speed = spawn.start_speed.
- Its goal is placed 'goal_ahead_distance' straight ahead of start heading,
  clamped within the world. Terminal when all reach, collision, or OOB.
- Observation per ego boat: 12 sectors × 8 features + 4 ego = 100 floats:
    For each sector (worst-risk target in that sector):
      [x_rel/rng, y_rel/rng, rel_brg/pi, dist/rng,
       tcpa/cpa_horizon, dcpa/rng, tgt_speed/max_speed, tgt_rel_h/pi]
    Ego tail: [x/world_w, y/world_h, u/max_speed, h/pi]
- Discrete action per step:
    idx = steer*3 + throttle
    steer    ∈ {0:straight, 1:right, 2:left}
    throttle ∈ {0:coast, 1:accelerate, 2:decelerate}

Rendering (optional, via pygame)
--------------------------------
- Hotkeys: H (HUD), G (grid), S (sectors), T (trails), ESC (quit).
- Call env.set_overlay({...}) to show live epsilon, episode, etc. on HUD.

Turn Session Logic (control-level, no physics)
----------------------------------------------
- If no session active and steer!=0 → start session toward target heading
  h_target = wrap(h + sign*turn_deg), sign=+1 (left), -1 (right).
- While active: advance heading at fixed yaw_rate_degps until within
  hysteresis_deg of target; then snap to target and clear the session.
- If allow_cancel=True: a new steer during a session *replaces* the session.

CPA: tcpa/dcpa used only for shaping/features (constant-velocity estimate).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False


# =============================
# Utils
# =============================
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def angle_deg(a: float) -> float:
    return a * 180.0 / math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x

def tcpa_dcpa(ax, ay, ah, aspd, bx, by, bh, bspd) -> Tuple[float, float]:
    """CPA under constant-velocity motion."""
    rvx = math.cos(bh) * bspd - math.cos(ah) * aspd
    rvy = math.sin(bh) * bspd - math.sin(ah) * aspd
    rx, ry = (bx - ax), (by - ay)
    rv2 = rvx*rvx + rvy*rvy
    if rv2 < 1e-9:
        return 0.0, math.hypot(rx, ry)
    tcpa = - (rx*rvx + ry*rvy) / rv2
    if tcpa < 0.0:
        tcpa = 0.0
    cx = rx + rvx * tcpa
    cy = ry + rvy * tcpa
    return tcpa, math.hypot(cx, cy)


# =============================
# Configs
# =============================
@dataclass
class BoatParams:
    # simple geometry for rendering/collision (meters)
    length: float = 6.0
    width:  float = 2.2

    # kinematic speed limits & linear accel model (m/s, m/s^2)
    max_speed:  float = 18.0
    min_speed:  float = 0.0
    accel_rate: float = 1.6
    decel_rate: float = 1.2

@dataclass
class TurnSessionConfig:
    # control-only "chunked turn"
    turn_deg: float = 15.0          # each session turns by this many degrees
    yaw_rate_degps: float = 45.0    # fixed yaw speed during session
    hysteresis_deg: float = 2.0     # deadband near target to avoid chatter
    allow_cancel: bool = False      # if True: new steer replaces current session
    passthrough_throttle: bool = True  # throttle accepted every step

@dataclass
class SpawnConfig:
    n_boats: int = 2
    start_speed: float = 0.0
    margin: float = 80.0
    goal_ahead_distance: float = 450.0
    goal_radius: float = 10.0

@dataclass
class EnvConfig:
    # world in meters
    world_w: int = 300
    world_h: int = 300

    dt: float = 0.05              # seconds per env step
    substeps: int = 1             # optional smoothing; 1 is fine here
    sensor_range: Optional[float] = None
    seed: Optional[int] = None

    # Rendering
    render: bool = False
    show_grid: bool = True
    show_sectors: bool = False
    show_trails: bool = True
    show_hud: bool = True
    pixels_per_meter: float = 3.0

    # Rewards
    progress_weight: float = 0.01
    living_penalty: float = -0.001
    goal_bonus: float = 6.0
    collision_penalty: float = -12.0
    oob_penalty: float = -3.0

    # CPA shaping (optional; set weight to 0 to disable)
    cpa_horizon: float = 60.0
    dcpa_scale: float = 120.0
    tcpa_decay: float = 20.0
    risk_weight: float = 0.10

    colregs_penalty: float = 0.05  # small nudge for basic stand-on behavior


# =============================
# Core Boat (kinematics + turn-session)
# =============================
class Boat:
    def __init__(self, boat_id: int, x: float, y: float, heading: float,
                 speed: float, kin: BoatParams, tcfg: TurnSessionConfig):
        self.id = boat_id
        self.x = float(x)
        self.y = float(y)
        self.h = float(heading)     # radians
        self.u = float(speed)       # m/s
        self.kin = kin
        self.tcfg = tcfg

        # discrete controls
        self.last_thr = 0           # -1 decel, 0 coast, +1 accel

        # turn session state
        self.session_active = False
        self.session_dir = 0        # +1 left, -1 right
        self.session_target = 0.0   # target heading (rad)

    @staticmethod
    def decode_action(a: int) -> Tuple[int, int]:
        steer = a // 3
        throttle = a % 3
        helm = (-1 if steer == 1 else +1 if steer == 2 else 0)   # right=-1, left=+1
        thr  = (-1 if throttle == 2 else +1 if throttle == 1 else 0)
        return helm, thr

    def start_turn_session(self, direction: int):
        # direction ∈ {-1, +1}
        dpsi = math.radians(self.tcfg.turn_deg) * direction
        self.session_dir = direction
        self.session_target = wrap_pi(self.h + dpsi)
        self.session_active = True

    def apply_action(self, a: int):
        helm, thr = self.decode_action(a)

        # Throttle passthrough (recommended)
        if self.tcfg.passthrough_throttle:
            self.last_thr = thr

        # Steering latching
        if not self.session_active:
            if helm != 0:
                self.start_turn_session(helm)
        else:
            if self.tcfg.allow_cancel and (helm != 0):
                self.start_turn_session(helm)  # replace current session
            # else: ignore steering until session completes

    def integrate(self, dt: float):
        # 1) Heading update from active session
        if self.session_active:
            rate = math.radians(self.tcfg.yaw_rate_degps)  # rad/s
            # shortest error
            err = wrap_pi(self.session_target - self.h)
            step = rate * dt
            if abs(err) <= (step + math.radians(self.tcfg.hysteresis_deg)):
                self.h = self.session_target
                self.session_active = False
                self.session_dir = 0
            else:
                self.h = wrap_pi(self.h + math.copysign(step, err))

        # 2) Speed update (linear accel/decel)
        if self.last_thr > 0:
            self.u += self.kin.accel_rate * dt
        elif self.last_thr < 0:
            self.u -= self.kin.decel_rate * dt
        # coast: keep speed (no passive drag)
        self.u = clamp(self.u, self.kin.min_speed, self.kin.max_speed)

        # 3) Position update (simple kinematics)
        self.x += math.cos(self.h) * self.u * dt
        self.y += math.sin(self.h) * self.u * dt


# =============================
# Environment
# =============================
class MultiBoatSectorsEnv:
    def __init__(self,
                 cfg: EnvConfig = EnvConfig(),
                 kin: BoatParams = BoatParams(),
                 tcfg: TurnSessionConfig = TurnSessionConfig(),
                 spawn: SpawnConfig = SpawnConfig()):
        self.cfg = cfg
        self.kin = kin
        self.tcfg = tcfg
        self.spawn = spawn
        self.rng = random.Random(cfg.seed)

        self.world_w = cfg.world_w
        self.world_h = cfg.world_h

        self.ships: List[Boat] = []
        self.goals: List[Tuple[float, float]] = []
        self.prev_goal_d: List[float] = []
        self.reached: List[bool] = []
        self.time = 0.0
        self.step_index = 0

        # rendering
        self._screen = None
        self._font = None
        self._clock = None
        self.ppm = float(cfg.pixels_per_meter)
        self.hud_on = bool(cfg.show_hud)
        self._overlay_extra: Dict = {}
        self._traces: List = []
        self._last_info: Dict = {}

        if self.cfg.render and HAS_PYGAME:
            self._setup_render()

    # ---------- Rendering ----------
    def _setup_render(self):
        pygame.init()
        width_px  = max(200, int(round(self.world_w * self.ppm)))
        height_px = max(200, int(round(self.world_h * self.ppm)))
        self._screen = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Multi-Boat (Turn Sessions) — Kinematic")
        self._font = pygame.font.Font(None, 18)
        self._clock = pygame.time.Clock()

    def enable_render(self):
        if not self._screen and HAS_PYGAME:
            self.cfg.render = True
            self._setup_render()

    def set_overlay(self, values: Optional[Dict]):
        self._overlay_extra = values or {}

    # world->screen
    def sx(self, x_m: float) -> int: return int(round(x_m * self.ppm))
    def sy(self, y_m: float) -> int: return int(round(y_m * self.ppm))

    def _handle_events(self):
        if not HAS_PYGAME:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_h: self.hud_on = not self.hud_on
                elif e.key == pygame.K_g: self.cfg.show_grid = not self.cfg.show_grid
                elif e.key == pygame.K_s: self.cfg.show_sectors = not self.cfg.show_sectors
                elif e.key == pygame.K_t: self.cfg.show_trails = not self.cfg.show_trails
                elif e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit

    def render(self):
        if not self._screen:
            return
        self._handle_events()
        surf = self._screen
        surf.fill((18, 52, 86))

        # border
        pygame.draw.rect(surf, (180, 180, 180),
                         (0, 0, self.sx(self.world_w), self.sy(self.world_h)), 2)

        # grid
        if self.cfg.show_grid:
            step_m = 100
            for xm in range(0, self.world_w + 1, step_m):
                x = self.sx(xm)
                pygame.draw.line(surf, (40, 70, 100), (x, 0), (x, self.sy(self.world_h)))
            for ym in range(0, self.world_h + 1, step_m):
                y = self.sy(ym)
                pygame.draw.line(surf, (40, 70, 100), (0, y), (self.sx(self.world_w), y))

        # goals
        goal_r_px = max(4, int(4 * self.ppm * 0.75))
        for i, (gx, gy) in enumerate(self.goals):
            pygame.draw.circle(surf, (250, 200, 50), (self.sx(gx), self.sy(gy)), goal_r_px)
            sh = self.ships[i]
            pygame.draw.line(surf, (200, 200, 80), (self.sx(sh.x), self.sy(sh.y)), (self.sx(gx), self.sy(gy)), 1)

        # trails
        if self._traces and self.cfg.show_trails:
            for i, sh in enumerate(self.ships):
                self._traces[i].append((self.sx(sh.x), self.sy(sh.y)))
                if len(self._traces[i]) > 2:
                    pygame.draw.lines(surf, (230, 230, 230), False, list(self._traces[i]), 1)

        # ships
        for sh in self.ships:
            self._draw_ship(surf, sh)
            if self.cfg.show_sectors:
                self._draw_sector_rays(surf, sh)

        self._draw_hud(surf)
        pygame.display.flip()
        self._clock.tick(60)

    def _draw_ship(self, surf, sh: Boat):
        Lm, Wm = self.kin.length, self.kin.width
        verts_local_m = [(+0.5*Lm, 0.0), (-0.5*Lm, -0.5*Wm), (-0.5*Lm, +0.5*Wm)]
        ch, shn = math.cos(sh.h), math.sin(sh.h)
        pts = []
        for vx_m, vy_m in verts_local_m:
            wx_m = sh.x + vx_m*ch - vy_m*shn
            wy_m = sh.y + vx_m*shn + vy_m*ch
            pts.append((self.sx(wx_m), self.sy(wy_m)))
        color = (90, 160, 255) if sh.id == 0 else (70, 200, 120)
        pygame.draw.polygon(surf, color, pts)
        # collision circle
        rad_px = max(1, int(round(self.collision_radius() * self.ppm)))
        pygame.draw.circle(surf, (255, 255, 255), (self.sx(sh.x), self.sy(sh.y)), rad_px, 1)
        label = self._font.render(f"{sh.id}", True, (255, 255, 255))
        surf.blit(label, (self.sx(sh.x) + 8, self.sy(sh.y) - 8))

    def _draw_sector_rays(self, surf, sh: Boat, n=12, ray_len_m: Optional[float] = None):
        Lm = ray_len_m or 320.0
        for k in range(n):
            ang = sh.h + 2.0 * math.pi * k / n
            x2 = self.sx(sh.x + Lm * math.cos(ang))
            y2 = self.sy(sh.y + Lm * math.sin(ang))
            pygame.draw.line(surf, (220, 220, 220), (self.sx(sh.x), self.sy(sh.y)), (x2, y2), 1)

    def _draw_hud(self, surf):
        if not self.hud_on:
            return
        font = pygame.font.Font(None, 18)
        pad, line = 8, 20
        fps = self._clock.get_fps() if self._clock else 0.0

        def fmt_reward(x):
            return f"{x:+.3f}" if isinstance(x, (int, float)) else "-"

        lines = [
            f"FPS {fps:5.1f}   step {self.step_index}   t {self.time:6.2f}s",
        ]
        if self._overlay_extra:
            ep = self._overlay_extra.get("episode", None)
            eps = self._overlay_extra.get("epsilon", None)
            gsteps = self._overlay_extra.get("global_steps", None)
            loss_ema = self._overlay_extra.get("loss_ema", None)
            if ep is not None: lines.append(f"ep {ep}   ε {eps:.3f}" if eps is not None else f"ep {ep}")
            if gsteps is not None: lines.append(f"gsteps {gsteps}")
            if isinstance(loss_ema, (int,float)): lines.append(f"loss_ema {loss_ema:.5f}")

        # per-boat readouts
        if self.ships:
            for i, sh in enumerate(self.ships):
                lines.append(f"B{i}: spd {sh.u:4.1f} m/s  hdg {math.degrees(sh.h):6.1f}°  "
                             f"turn={'ON' if sh.session_active else 'OFF'}")
                gx, gy = self.goals[i]
                gd = math.hypot(gx - sh.x, gy - sh.y)
                r = None
                if self._last_info:
                    rew = self._last_info.get("rewards", None)
                    if isinstance(rew, (list, tuple)) and i < len(rew):
                        r = rew[i]
                lines.append(f"  goal_d {gd:6.1f} m   reached={self.reached[i] if i < len(self.reached) else False}   r={fmt_reward(r)}")

        reason = self._last_info.get("reason", "")
        if reason: lines.append(f"reason: {reason}")

        w = max(font.size(ln)[0] for ln in lines) + 2*pad
        h = len(lines)*line + 2*pad
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        y = pad
        for ln in lines:
            img = font.render(ln, True, (240, 240, 240))
            panel.blit(img, (pad, y))
            y += line
        surf.blit(panel, (10, 10))

    # ---------- Core ----------
    def collision_radius(self) -> float:
        return self.kin.length * 0.8

    def _outside(self, sh: Boat) -> bool:
        return not (0 <= sh.x <= self.world_w and 0 <= sh.y <= self.world_h)

    def reset(self, seed: Optional[int] = None) -> List[np.ndarray]:
        if seed is not None:
            self.rng.seed(seed)

        self.ships = []
        self.goals = []
        self.prev_goal_d = []
        self.reached = [False] * self.spawn.n_boats
        self.time = 0.0
        self.step_index = 0
        self._last_info = {}

        margin = self.spawn.margin
        for i in range(self.spawn.n_boats):
            # ensure separated starts
            for _attempt in range(1000):
                x = self.rng.uniform(margin, self.world_w - margin)
                y = self.rng.uniform(margin, self.world_h - margin)
                h = self.rng.uniform(-math.pi, math.pi)
                ok = True
                for sh in self.ships:
                    if (sh.x - x)**2 + (sh.y - y)**2 < (4 * self.collision_radius())**2:
                        ok = False; break
                if ok:
                    boat = Boat(i, x, y, h, self.spawn.start_speed, self.kin, self.tcfg)
                    self.ships.append(boat)
                    # goal straight ahead, clamped
                    gx = x + self.spawn.goal_ahead_distance * math.cos(h)
                    gy = y + self.spawn.goal_ahead_distance * math.sin(h)
                    gx = clamp(gx, 20, self.world_w - 20)
                    gy = clamp(gy, 20, self.world_h - 20)
                    self.goals.append((gx, gy))
                    self.prev_goal_d.append(math.hypot(gx - x, gy - y))
                    break

        # trails
        from collections import deque
        self._traces = [deque(maxlen=600) for _ in self.ships]

        return self.get_obs_all()

    # ---- Observations (100-d per boat) ----
    def get_obs(self, i: int) -> np.ndarray:
        ego = self.ships[i]
        others = [s for s in self.ships if s is not ego]

        N_SECT = 12
        buckets: Dict[int, List[Tuple[float, dict]]] = {k: [] for k in range(N_SECT)}

        diag = math.hypot(self.world_w, self.world_h)
        rng = self.cfg.sensor_range or diag

        for tgt in others:
            dx, dy = tgt.x - ego.x, tgt.y - ego.y
            dist = math.hypot(dx, dy)
            if self.cfg.sensor_range and dist > self.cfg.sensor_range:
                continue

            ch, shn = math.cos(ego.h), math.sin(ego.h)
            x_rel =  ch * dx + shn * dy
            y_rel = -shn * dx + ch * dy
            rel_brg = math.atan2(y_rel, x_rel)

            rel_deg = (angle_deg(rel_brg) + 360.0) % 360.0
            sector = int(rel_deg // (360.0 / N_SECT))

            tcpa, dcpa = tcpa_dcpa(ego.x, ego.y, ego.h, ego.u, tgt.x, tgt.y, tgt.h, tgt.u)

            # "risk" score (more negative => higher risk) to pick worst target in sector
            w_tcpa = math.exp(-tcpa / max(1e-6, self.cfg.tcpa_decay))
            w_dcpa = math.exp(-dcpa / max(1e-6, self.cfg.dcpa_scale))
            score = -(w_tcpa * w_dcpa)

            buckets[sector].append((score, dict(
                x_rel=x_rel, y_rel=y_rel,
                rel_brg=rel_brg, dist=dist,
                tcpa=tcpa, dcpa=dcpa,
                tgt_speed=tgt.u,
                tgt_rel_h=wrap_pi(tgt.h - ego.h),
            )))

        feats = []
        for k in range(N_SECT):
            if buckets[k]:
                buckets[k].sort(key=lambda t: t[0])
                d = buckets[k][0][1]
                feats.extend([
                    d["x_rel"]/rng, d["y_rel"]/rng,
                    d["rel_brg"]/math.pi,
                    d["dist"]/rng,
                    clamp(d["tcpa"], 0.0, self.cfg.cpa_horizon)/self.cfg.cpa_horizon,
                    clamp(d["dcpa"], 0.0, rng)/rng,
                    d["tgt_speed"]/self.kin.max_speed,
                    d["tgt_rel_h"]/math.pi
                ])
            else:
                feats.extend([0.0]*8)

        feats.extend([
            ego.x / self.world_w,
            ego.y / self.world_h,
            ego.u / self.kin.max_speed,
            ego.h / math.pi,
        ])
        assert len(feats) == 100
        return np.asarray(feats, dtype=np.float32)

    def get_obs_all(self) -> List[np.ndarray]:
        return [self.get_obs(i) for i in range(len(self.ships))]

    # ---- Step ----
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        assert len(actions) == len(self.ships), "actions length must equal number of boats"

        # remember previous goal distances
        d_prev = [self.prev_goal_d[i] for i in range(len(self.ships))]

        # apply commands
        for sh, a in zip(self.ships, actions):
            sh.apply_action(a)

        # integrate kinematics
        dt = self.cfg.dt / max(1, self.cfg.substeps)
        for _ in range(self.cfg.substeps):
            for sh in self.ships:
                sh.integrate(dt)

        self.time += self.cfg.dt
        self.step_index += 1

        # terminals
        done = False
        reason = ""

        # collision: any pair within 2R
        R = self.collision_radius()
        R2 = (2*R)**2
        for i in range(len(self.ships)):
            for j in range(i+1, len(self.ships)):
                dx = self.ships[i].x - self.ships[j].x
                dy = self.ships[i].y - self.ships[j].y
                if dx*dx + dy*dy <= R2:
                    done, reason = True, "collision"
                    break
            if done: break

        # out of bounds
        if not done:
            for sh in self.ships:
                if self._outside(sh):
                    done, reason = True, "out_of_bounds"
                    break

        # goal status
        d_now = []
        just_arrived = [False]*len(self.ships)
        for i, sh in enumerate(self.ships):
            gx, gy = self.goals[i]
            d = math.hypot(gx - sh.x, gy - sh.y)
            d_now.append(d)
            if not self.reached[i] and d <= self.spawn.goal_radius:
                self.reached[i] = True
                just_arrived[i] = True

        if not done and all(self.reached):
            done, reason = True, "goals_reached"

        # rewards
        rewards = []
        for i in range(len(self.ships)):
            r = 0.0
            r += self.cfg.progress_weight * (d_prev[i] - d_now[i])
            r += self.cfg.living_penalty
            if just_arrived[i] and (d_prev[i] > self.spawn.goal_radius):
                r += self.cfg.goal_bonus

            # CPA shaping (optional)
            if self.cfg.risk_weight > 0.0:
                ego = self.ships[i]
                # compute worst w over other targets
                w_max = 0.0
                for j, tgt in enumerate(self.ships):
                    if j == i: continue
                    tcpa, dcpa = tcpa_dcpa(ego.x, ego.y, ego.h, ego.u, tgt.x, tgt.y, tgt.h, tgt.u)
                    if 0.0 <= tcpa <= self.cfg.cpa_horizon:
                        w = math.exp(-tcpa / self.cfg.tcpa_decay) * math.exp(-dcpa / self.cfg.dcpa_scale)
                        w_max = max(w_max, w)
                        # simple COLREGs nudge: if target on starboard forward quarter and we are turning to PORT
                        # we can't see "turning dir" here cleanly; as a light nudge, use heading-rate sign via session
                        if getattr(ego, "session_active", False) and ego.session_dir > 0:
                            # turning left (porting)
                            # compute relative bearing in ego frame
                            rel_brg = math.atan2(
                                -(math.sin(ego.h) * (tgt.x - ego.x) - math.cos(ego.h) * (tgt.y - ego.y)),
                                (math.cos(ego.h) * (tgt.x - ego.x) + math.sin(ego.h) * (tgt.y - ego.y))
                            )
                            rel_deg = (angle_deg(rel_brg) + 360.0) % 360.0
                            if 0.0 <= rel_deg <= 112.5:
                                r -= self.cfg.colregs_penalty * w
                r -= self.cfg.risk_weight * w_max

            if reason == "collision":
                r += self.cfg.collision_penalty
            elif reason == "out_of_bounds":
                r += self.cfg.oob_penalty

            rewards.append(float(r))

        # update prev distances
        for i in range(len(self.ships)):
            self.prev_goal_d[i] = d_now[i]

        info = dict(rewards=tuple(rewards), reason=reason, reached=tuple(self.reached))
        self._last_info = info
        return self.get_obs_all(), rewards, done, info
