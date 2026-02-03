import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm

# --- Data Loading and Preparation
df = pd.read_csv("experiments/gpt_experiments/moderate/moderate_prompt_multi2_experiment_4.csv")
df = df.sort_values("time_s")

times = np.unique(df["time_s"].values)
vessel_ids = np.unique(df["vessel_id"].values)

# Create a 3D array for coordinates (num_vessels, num_times, 2)
coords = np.zeros((len(vessel_ids), len(times), 2))

# Store goals if available
has_goals = ("goal_x" in df.columns and "goal_y" in df.columns)
if has_goals:
    goals = np.zeros((len(vessel_ids), 2))

for i, vid in enumerate(vessel_ids):
    # Get data for this vessel, indexed by time
    sub = df[df["vessel_id"] == vid].set_index("time_s")[["x", "y"]]
    # Reindex to fill missing time steps with the nearest available data
    re = sub.reindex(times, method="nearest")
    coords[i, :, 0] = re["x"].values
    coords[i, :, 1] = re["y"].values

    # Store goal if available
    if has_goals:
        gxs = df[df["vessel_id"] == vid]["goal_x"].dropna().unique()
        gys = df[df["vessel_id"] == vid]["goal_y"].dropna().unique()
        if len(gxs) and len(gys):
            goals[i, 0] = gxs[0]
            goals[i, 1] = gys[0]


fig, ax = plt.subplots(figsize=(3*1.9685, 3*1.9685), dpi=100)
plt.subplots_adjust(bottom=0.25)

# Define colors for vessels
# Using a colormap for distinct colors
colors = [cm.viridis(i / len(vessel_ids)) for i in range(len(vessel_ids))]

# plot full tracks in light dashed
for i, vid in enumerate(vessel_ids):
    ax.plot(coords[i, :, 0], coords[i, :, 1], "--", color=colors[i], alpha=0.7, label=f"V{i + 1}", linewidth=2.0)

# plot goals
if has_goals:
    for i in range(len(vessel_ids)):
        ax.scatter(goals[i, 0], goals[i, 1], marker="*", s=150, color=colors[i], edgecolor="k",
                   zorder=2)

ax.invert_yaxis()
ax.set_aspect("auto")
ax.set_xlabel("X", fontweight='bold')
ax.set_ylabel("Y", fontweight='bold')
ax.set_title("Vessel Trajectories", fontweight='bold')
legend_obj = ax.legend(loc='upper right')

# Make the legend labels bold
for text in legend_obj.get_texts():
    text.set_fontweight('bold')

vessel_patches = []

# Define triangle dimensions here
triangle_head_width = 7
triangle_head_length = 20

# The vector length used to define direction (can be small when lw=0)
arrow_vector_length = 0.1

for i, vid in enumerate(vessel_ids):
    # Create a FancyArrowPatch using the defined dimensions
    # -|> style creates a triangle head
    patch = FancyArrowPatch((0, 0), (arrow_vector_length, 0),
                            arrowstyle=f"-|>,head_width={triangle_head_width},head_length={triangle_head_length}",
                            color=colors[i],
                            mutation_scale=1,  # Controls the size scaling (1 is default) - you could also scale here
                            lw=0,  # Line width of the arrow body (0 makes it just a triangle)
                            zorder=3)  # Ensure it's drawn on top
    ax.add_patch(patch)  # Add the patch to the axes
    vessel_patches.append(patch)


# In the update function, the logic remains the same:
def update(frame_idx):
    prev_frame_idx = max(0, frame_idx - 1)

    for i, patch in enumerate(vessel_patches):
        x_curr, y_curr = coords[i, frame_idx]
        x_prev, y_prev = coords[i, prev_frame_idx]

        dx = x_curr - x_prev
        dy = y_curr - y_prev

        posA = (x_curr, y_curr)

        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist > 0:
            unit_dx = dx / dist
            unit_dy = dy / dist
            posB = (x_curr + unit_dx * arrow_vector_length, y_curr + unit_dy * arrow_vector_length)
        else:
            posB = (x_curr, y_curr)
        patch.set_positions(posA, posB)

    return vessel_patches


# --- Slider Setup ---
ax_slider = plt.axes([0.15, 0.10, 0.70, 0.03])
slider = Slider(ax_slider, "Time (s)", times[0], times[-1], valinit=times[0],
                valstep=times[1] - times[0] if len(times) > 1 else 1)


def on_slider(val):
    idx = int(np.argmin(np.abs(times - val)))
    # Update the plot to that time step
    update(idx)
    fig.canvas.draw_idle()


slider.on_changed(on_slider)

ani = FuncAnimation(
    fig,
    update,
    frames=len(times),
    interval=200,
    blit=True,  # Optimize drawing by only redrawing changed artists
    repeat=False  # Don't loop the animation
)

# To save the animation
# ani.save('vessel_movement.gif', writer='pillow', fps=5)
# ani.save('vessel_movement.mp4', writer='ffmpeg', fps=5)

plt.show()
