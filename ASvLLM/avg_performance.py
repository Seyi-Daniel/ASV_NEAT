import pandas as pd
import numpy as np
import os
import json
import re
from math import sqrt
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Parameters
collision_radius = 20
goal_radius = 20

# Paths to your 10 CSV files
csv_files = [os.path.join("experiments","deepseek_experiments", "minimal", f"minimal_prompt_head_on_experiment_{i}.csv") for i in range(1, 11)]

# Storage for each run’s metrics
all_success_rates = []
all_collision_rates = []
all_avg_path_lengths = []
all_avg_times_to_goal = []
all_total_collisions = []

def compute_metrics(df):
    def compute_reached_goal_once(subdf):
        subdf = subdf.sort_values("time_s").copy()
        subdf["reached_goal_computed"] = False
        for idx, row in subdf.iterrows():
            dist = sqrt((row["x"] - row["goal_x"])**2 + (row["y"] - row["goal_y"])**2)
            if dist < goal_radius:
                subdf.at[idx, "reached_goal_computed"] = True
                break
        return subdf

    def detect_collisions(time_df):
        vessels = time_df[["vessel_id", "x", "y"]].values
        count = 0
        for (id1, x1, y1), (id2, x2, y2) in combinations(vessels, 2):
            if sqrt((x1 - x2)**2 + (y1 - y2)**2) < collision_radius:
                count += 1
        return count

    def compute_path(subdf):
        subdf = subdf.sort_values("time_s")
        dx = subdf["x"].diff()
        dy = subdf["y"].diff()
        return np.sqrt(dx**2 + dy**2).sum()

    def compute_time_to_goal(subdf):
        subdf = subdf.sort_values("time_s")
        for _, row in subdf.iterrows():
            dist = sqrt((row["x"] - row["goal_x"])**2 + (row["y"] - row["goal_y"])**2)
            if dist < goal_radius:
                return row["time_s"]
        return np.nan

    # Compute all metrics
    df = df.groupby("vessel_id", group_keys=False).apply(compute_reached_goal_once)
    success_rate = df.groupby("vessel_id")["reached_goal_computed"].max().mean() * 100
    collision_events = df.groupby("time_s").apply(detect_collisions)
    total_collisions = collision_events.sum()
    collision_rate = (total_collisions / df["time_s"].nunique()) * 100
    avg_path_length = df.groupby("vessel_id").apply(compute_path).mean()
    avg_time_to_goal = df.groupby("vessel_id").apply(compute_time_to_goal).dropna().mean()

    return success_rate, collision_rate, avg_path_length, avg_time_to_goal, total_collisions

# Loop through all files and compute metrics
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    success, collision, path_len, time_goal, total_coll = compute_metrics(df)
    print(f"[Run {i+1}] Success: {success:.2f}%, Collision Rate: {collision:.2f}%, Avg Path: {path_len:.2f}, Time to Goal: {time_goal:.2f}, Total Collisions: {total_coll}")

    all_success_rates.append(success)
    all_collision_rates.append(collision)
    all_avg_path_lengths.append(path_len)
    all_avg_times_to_goal.append(time_goal)
    all_total_collisions.append(total_coll)

# Aggregate across all files
print("\n=== Averaged Performance Metrics over 10 files ===")
print(f"Average Success Rate           : {np.mean(all_success_rates):.2f}%")
print(f"Average Collision Rate         : {np.mean(all_collision_rates):.2f}%")
print(f"Average Path Length            : {np.mean(all_avg_path_lengths):.2f} units")
print(f"Average Time to Goal           : {np.mean(all_avg_times_to_goal):.2f} seconds")
print(f"Average Number of Collisions   : {np.sum(all_total_collisions) / len(csv_files):.2f}")
print(f"Total Collisions across files  : {np.sum(all_total_collisions)}")
