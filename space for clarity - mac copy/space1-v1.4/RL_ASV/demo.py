import argparse
import math
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from channel_crossing_env import ChannelCrossingEnv  # or from env import ChannelCrossingEnv


class DQN_CNN(nn.Module):
    """
    Atari-style convolutional Q-network.
    Expects input of shape (B, 4, 84, 84) — a stack of 4 grayscale frames.
    """
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 512)
        self.head  = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.head(x)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Convert HxWx3 RGB frame to 84x84 normalized grayscale float32 in [0,1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def wait_if_paused(env):
    """Keep pygame window responsive when paused."""
    if getattr(env, "render_mode", None) == "human":
        env.render()
        while getattr(env, "paused", False):
            env.render()


def run_episode_with_optional_t1_mask(env: ChannelCrossingEnv,
                                      policy_net: DQN_CNN,
                                      device,
                                      seed: int,
                                      mask: Optional[int]):
    """
    If mask is not None:
      - Choose a0 from s0=[f0]*4
      - step once (t=1)
      - delete per mask BEFORE the next render (so s1 uses f1(modified))
      - roll out to the end as normal
    If mask is None, just run a normal episode.
    """
    # Reset to deterministic scenario
    env.reset(seed)

    # Build s0 = [f0]*4
    f0 = env.render()
    motion_frames = deque(maxlen=4)
    g0 = preprocess(f0)
    for _ in range(4):
        motion_frames.append(g0)
    state = np.stack(motion_frames, axis=0)

    done = False
    total_reward = 0.0
    collided = False
    prev_x, prev_y = env.agent.x, env.agent.y

    # Decide first action from s0
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        a0 = int(policy_net(state_tensor).argmax(dim=1).item())

    # Take step 1 (t=1)
    obs, reward, terminated, truncated, _ = env.step(a0)
    total_reward += reward
    done = terminated or truncated

    # **t=1 deletion happens here (before the next render)**
    if mask is not None:
        env.delete_traffic_mask(mask, permanent=True)

    # Now render → this frame is f1(modified) if mask was applied
    frame = env.render()
    motion_frames.append(preprocess(frame))
    state = np.stack(motion_frames, axis=0)

    # Continue as normal
    while not done:
        wait_if_paused(env)

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = int(policy_net(state_tensor).argmax(dim=1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Collision diagnostic (ignores deleted/inactive)
        if any((not getattr(tb, "deleted", False)) and getattr(tb, "active", True)
               and env._check_halo_overlap(env.agent, tb)
               for tb in env.traffic):
            collided = True

        # Track distance (optional)
        curr_x, curr_y = env.agent.x, env.agent.y
        _seg = math.hypot(curr_x - prev_x, curr_y - prev_y)
        prev_x, prev_y = curr_x, curr_y

        frame = env.render()
        motion_frames.append(preprocess(frame))
        state = np.stack(motion_frames, axis=0)

    # Outcome summary
    dist_to_goal = math.hypot(env.goal_x - env.agent.x, env.goal_y - env.agent.y)
    reached_goal = (dist_to_goal < 5.0)
    outcome = "SUCCESS" if reached_goal else ("COLLISION" if collided else "TIMEOUT")

    return {
        "mask": mask,
        "deleted_indices": ([] if mask is None else [i for i in range(env.n_traffic) if (mask >> i) & 1]),
        "first_action": a0,
        "total_reward": total_reward,
        "outcome": outcome,
        "steps": env.steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ChannelCrossingEnv with optional t=1 deletion sweep")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the .pth checkpoint file')
    parser.add_argument('--render-mode', type=str, choices=['human', 'rgb_array'], default='human')
    parser.add_argument('--window-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1, help='Static seed for the scenario (use one seed for all 64 runs)')
    parser.add_argument('--t1-sweep', action='store_true', help='Run 64 combinations of deletions at t=1')
    parser.add_argument('--mask', type=int, default=None, help='Run a single mask (0..63) at t=1; overrides --episodes loop')
    parser.add_argument('--episodes', type=int, default=10, help='Normal episode count (ignored by --t1-sweep and --mask)')
    args = parser.parse_args()

    env = ChannelCrossingEnv(render_mode=args.render_mode,
                             window_name=(args.window_name or "Channel Crossing (Demo)"))
    env = env.unwrapped
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN_CNN(num_actions).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    # --- Mode 1: full 64-way sweep at t=1 (static seed) ---
    if args.t1_sweep and args.mask is None:
        results = []
        n = env.n_traffic
        print(f"\nRunning t=1 sweep for seed={args.seed} over {1<<n} masks...")
        for mask in range(1 << n):
            res = run_episode_with_optional_t1_mask(env, policy_net, device, seed=args.seed, mask=mask)
            print(f"mask={mask:02d} delete={res['deleted_indices']}  a0={res['first_action']}  "
                  f"steps={res['steps']}  R={res['total_reward']:.2f}  outcome={res['outcome']}")
            results.append(res)

        succ = sum(1 for r in results if r["outcome"] == "SUCCESS")
        coll = sum(1 for r in results if r["outcome"] == "COLLISION")
        print("\n=== t=1 sweep summary ===")
        print(f"  Successes : {succ} / {len(results)}")
        print(f"  Collisions: {coll} / {len(results)}")
        env.close()
        return

    # --- Mode 2: single-mask run (handy for quick checks) ---
    if args.mask is not None:
        res = run_episode_with_optional_t1_mask(env, policy_net, device, seed=args.seed, mask=args.mask)
        print(f"\nSingle mask run: mask={args.mask} delete={res['deleted_indices']}  "
              f"a0={res['first_action']}  steps={res['steps']}  R={res['total_reward']:.2f}  outcome={res['outcome']}")
        env.close()
        return

    # --- Mode 3: your original multi-episode loop (unchanged behavior) ---
    success_count = 0
    collision_count = 0
    total_distance_success = 0.0

    print(
        "\nControls: P=pause/resume. While paused: "
        "Left-click=select ship, Ctrl+A=select all, D/Delete=remove, Right-click=clear, Esc=unselect."
    )

    for ep in range(1, args.episodes + 1):
        # Keep your old per-episode seeding if you're not sweeping masks
        env.reset(ep)

        prev_x, prev_y = env.agent.x, env.agent.y
        distance_travelled = 0.0
        collided = False

        # Build s0
        frame = env.render()
        motion_frames = deque(maxlen=4)
        g = preprocess(frame)
        for _ in range(4):
            motion_frames.append(g)
        state = np.stack(motion_frames, axis=0)

        done = False
        total_reward = 0.0
        while not done:
            wait_if_paused(env)

            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(policy_net(state_tensor).argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)

            curr_x, curr_y = env.agent.x, env.agent.y
            distance_travelled += math.hypot(curr_x - prev_x, curr_y - prev_y)
            prev_x, prev_y = curr_x, curr_y

            if any((not getattr(tb, "deleted", False)) and getattr(tb, "active", True)
                   and env._check_halo_overlap(env.agent, tb)
                   for tb in env.traffic):
                collided = True

            done = terminated or truncated
            total_reward += reward

            frame = env.render()
            g = preprocess(frame)
            motion_frames.append(g)
            state = np.stack(motion_frames, axis=0)

        dist_to_goal = math.hypot(env.goal_x - env.agent.x, env.goal_y - env.agent.y)
        reached_goal = (dist_to_goal < 5.0)
        crossed_line = (env.agent.y >= env.channel_high)

        if collided:
            collision_count += 1
            outcome = "COLLISION"
        elif reached_goal or crossed_line:
            success_count += 1
            total_distance_success += distance_travelled
            outcome = "SUCCESS"
        else:
            outcome = "OTHER"

        print(f"Episode {ep}: reward {total_reward:.2f}   {outcome}")

    env.close()

    avg_distance = (total_distance_success / success_count) if success_count else 0.0
    print("\n=== Summary over {} episodes ===".format(args.episodes))
    print(f"  Successes: {success_count}")
    print(f"  Collisions: {collision_count}")
    print(f"  Avg. distance on successes: {avg_distance:.2f}")


if __name__ == '__main__':
    main()
