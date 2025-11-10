import argparse
import math
import os
import random
from collections import deque

import cv2  # for grayscale & resize
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from channel_crossing_env import ChannelCrossingEnv

# ----- Direct configuration (set these variables) -----
EPISODES      = 100         # number of episodes to run
RENDER_MODE   = "human"     # "human" or "rgb_array"
# ----------------------------------------------------

class DQN_CNN(nn.Module):
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
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser(description="Run a trained DDQN model")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the .pth checkpoint file')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to run')
    parser.add_argument('--render-mode', type=str, choices=['human', 'rgb_array'],
                        default=None, help='Render mode for the environment')
    parser.add_argument('--window-name', type=str,
                        default=None, help='Name of the Simulation window')
    args = parser.parse_args()

    model_path = args.model_path
    num_episodes = args.episodes if args.episodes is not None else EPISODES
    render_mode = args.render_mode if args.render_mode is not None else RENDER_MODE
    window_name = args.window_name if args.window_name is not None else model_path

    env = ChannelCrossingEnv(render_mode=render_mode, window_name=window_name)
    env = env.unwrapped
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN_CNN(num_actions).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint['model_state'])
    policy_net.eval()

    success_goal_count = 0
    success_cross_count = 0
    fail_collision_count = 0
    fail_oob_count = 0
    total_distance_success = 0.0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(ep)
        prev_x, prev_y = env.agent.x, env.agent.y
        distance_travelled = 0.0
        collided = False
        crossed_channel = False
        steps = 0

        frame = env.render()
        motion_frames = deque(maxlen=4)
        gray = preprocess(frame)
        for _ in range(4):
            motion_frames.append(gray)
        state = np.stack(motion_frames, axis=0)

        done = False
        total_reward = 0.0
        while not done:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                qvals = policy_net(state_tensor)
                action = int(qvals.argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)

            curr_x, curr_y = env.agent.x, env.agent.y
            distance_travelled += math.hypot(curr_x - prev_x, curr_y - prev_y)
            prev_x, prev_y = curr_x, curr_y

            if any(env._check_halo_overlap(env.agent, tb) for tb in env.traffic):
                collided = True

            if env.agent.y >= env.channel_high:
                crossed_channel = True
            elif crossed_channel and env.agent.y < env.channel_high:
                crossed_channel = False

            done = terminated or truncated
            total_reward += reward
            steps += 1

            frame = env.render()
            gray = preprocess(frame)
            motion_frames.append(gray)
            state = np.stack(motion_frames, axis=0)

        dist_to_goal = math.hypot(env.goal_x - env.agent.x, env.goal_y - env.agent.y)
        reached_goal  = (dist_to_goal < 5.0)
        out_of_bounds = (env.agent.x < 0 or env.agent.x > env.width or
                         env.agent.y < 0 or env.agent.y > env.height)

        if collided:
            fail_collision_count += 1
            outcome = "Fail (Collision)"
        elif reached_goal:
            success_goal_count += 1
            total_distance_success += distance_travelled
            outcome = "Success (Crossed Channel & Reached Goal)"
        elif crossed_channel:
            success_cross_count += 1
            total_distance_success += distance_travelled
            outcome = "Success (Crossed Channel & Missed Goal)"
        elif out_of_bounds:
            fail_oob_count += 1
            outcome = "Fail (Out-of-bounds)"
        else:
            outcome = "Other"

        print(f"Episode {ep}: reward {total_reward:.2f}   {outcome}")

    env.close()

    successes_total = success_goal_count + success_cross_count
    avg_distance = (total_distance_success / successes_total) if successes_total else 0.0

    print(f"\n=== Summary over {num_episodes} episodes ===")
    print(f"  Success (Crossed Channel & Reached Goal): {success_goal_count}")
    print(f"  Success (Crossed Channel & Missed Goal): {success_cross_count}")
    print(f"  Fail (Collision): {fail_collision_count}")
    print(f"  Fail (Out-of-bounds): {fail_oob_count}")
    print(f"  Avg. distance on successes: {avg_distance:.2f}")

    # === Main stacked outcomes figure with percentages ===
    s_goal = success_goal_count
    s_cross = success_cross_count
    f_collision = fail_collision_count
    f_oob = fail_oob_count
    total = s_goal + s_cross + f_collision + f_oob

    # Compute percentages for legend labels
    def pct(v):
        return f" ({(v/total*100):.1f}%)" if total > 0 else ""

    plt.figure(figsize=(9, 5))

    # Colors and data
    c_goal = 'tab:blue'
    c_cross = '#9ecae1'
    c_collision = 'tab:red'
    c_oob = '#fcaeae'

    plt.bar(["Success"], [s_goal], label=f"Success (Crossed Channel & Reached Goal){pct(s_goal)}", color=c_goal)
    plt.bar(["Success"], [s_cross], bottom=[s_goal], label=f"Success (Crossed Channel & Missed Goal){pct(s_cross)}", color=c_cross)
    plt.bar(["Failure"], [f_collision], label=f"Fail (Collision){pct(f_collision)}", color=c_collision)
    plt.bar(["Failure"], [f_oob], bottom=[f_collision], label=f"Fail (Out-of-bounds){pct(f_oob)}", color=c_oob)

    plt.ylabel("Count")
    plt.title(
        f"Imitation Learning (IL) Outcomes — "
        f"Avg success distance: {avg_distance:.2f}\n"
        f"Over {num_episodes} scenarios, (Facing Upwards, Far from Edges)",
        fontsize=12,
        pad=20
    )

    # Legend styling
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper center",
            bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("episode_outcomes6.png", bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
