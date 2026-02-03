import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/claude_experiments/detailed/detailed_prompt_multi2_experiment_9.csv")

fig, ax = plt.subplots()

for vid, group in df.groupby("vessel_id"):
    ax.plot(group["x"], group["y"], label=f"Vessel {vid}")
    # mark start
    ax.scatter(group.iloc[0]["x"], group.iloc[0]["y"], marker="o", s=50, edgecolor="k")
    # mark goal
    gx, gy = group.iloc[0]["goal_x"], group.iloc[0]["goal_y"]
    ax.scatter(gx, gy, marker="*", s=100, edgecolor="r")

# flip matplotlib's y-axis so it matches pygame coordinates
ax.invert_yaxis()

ax.set_aspect("auto")
ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.set_title("Vessel Trajectories")
ax.legend()
plt.show()
