import math
import json  # <-- Added for formatting history


# calculate_relative_bearing is unchanged...
def calculate_relative_bearing(own_ship, target_ship):
    """Calculates bearing of target relative to own ship's heading (0=ahead, 90=starboard/CW)."""
    dx = target_ship.x - own_ship.x
    dy = target_ship.y - own_ship.y
    world_angle_rad = math.atan2(-dy, dx)
    own_heading_rad = (math.pi / 2 - own_ship.heading) % (2 * math.pi)
    relative_angle_rad = world_angle_rad - own_heading_rad
    relative_bearing_rad = (-relative_angle_rad) % (2 * math.pi)
    return math.degrees(relative_bearing_rad)


# --- UPDATED FUNCTION SIGNATURE and PROMPT STRING ---
def generate_vessel_prompt(vessels_to_describe, all_context_vessels, pixels_per_km=1, previous_vessel_data_list=None,
                           previous_responses=None):
    """
    Generates a standard prompt, including structured historical context.
    """
    prompt = ("""
"You are a ship navigation expert system following COLREGs.
Analyze the provided image and the **current** vessel data below. Also consider the **recent history** of the situation and decisions made.

"""
              )

    # --- NEW, ROBUST HISTORY SECTION ---
    if previous_vessel_data_list and previous_responses and len(previous_vessel_data_list) == len(previous_responses):
        prompt += "**Recent History (Most recent first):**\n"
        for i in range(len(previous_vessel_data_list) - 1, -1, -1):
            history_step = len(previous_vessel_data_list) - i
            prompt += f"--- History Step T-{history_step} ---\n"

            prompt += f"Vessel States (T-{history_step}):\n"
            for vessel_state in previous_vessel_data_list[i]:
                prompt += f"  - id: {vessel_state['id']}, pos: {vessel_state['pos']}, heading: {vessel_state['heading_deg']}°, speed: {vessel_state['speed_kmh']} km/h\n"

            prompt += f"Decision Made (T-{history_step}):\n"
            for decision in previous_responses[i]:
                prompt += f"  - id: {decision.get('id', 'N/A')}, situation: {decision.get('situation', 'N/A')}, role: {decision.get('role', 'N/A')}, action: {decision.get('action', 'N/A')}\n"
            prompt += "\n"
        prompt += "---\n\n"
    # --- END HISTORY SECTION ---

    prompt += "**Current Situation Analysis Task:**\n"
    # Append your original standard instructions
    prompt += """
**Analysis Instructions:**
1.  **Overtaking vs. Head-on:** An 'Overtaking' situation occurs when one vessel (faster) approaches another from a relative bearing near 0°, while the other vessel (slower) sees the first at a relative bearing near 180°. A 'Head-on' situation involves two vessels on reciprocal courses.
2.  **Crossing:** A 'Crossing' situation occurs when two power-driven vessels' paths cross. The vessel which has the other on her own starboard side is the 'Give-way' vessel.
3.  **Roles:** Assign 'Give-way' (must take action) or 'Stand-on' (must maintain course and speed).

**Output Format Requirements:**
Respond ONLY with a JSON array of objects {id, situation, role, action} for each vessel listed in the main '--- Current Vessel Data ---' section.
The 'action' value MUST be one of the following exact phrases:
- 'Alter course to starboard'
- 'Alter course to port'
- 'Reduce speed'
- 'Pass astern'
- 'Accelerate'
- 'Maintain course and speed'

--- Current Vessel Data ---"
"""

    # --- Data Population Logic (Identical to your original) ---
    for v in vessels_to_describe:
        speed_kmh = (v.speed / pixels_per_km * 3600) if pixels_per_km > 0 else 0
        prompt += f"\n- id: {id(v)}, position: ({v.x:.1f}, {v.y:.1f}) px, heading: {math.degrees(v.heading):.1f}°, speed: {speed_kmh:.1f} km/h"
        prompt += "\n  Other vessels:"
        other_vessels_found = False
        for o in all_context_vessels:
            if o is v: continue
            dx, dy = o.x - v.x, o.y - v.y
            dist_km = math.hypot(dx, dy) / pixels_per_km if pixels_per_km > 0 else 0
            rb = calculate_relative_bearing(v, o)
            sp = (o.speed / pixels_per_km * 3600) if pixels_per_km > 0 else 0
            prompt += (
                f"\n    - id: {id(o)}, pos: ({o.x:.1f},{o.y:.1f}) px, heading: {math.degrees(o.heading):.1f}°,"
                f" speed: {sp:.1f} km/h, dist: {dist_km:.3f} km, relBearing: {rb:.1f}°"
            )
            other_vessels_found = True

        if not other_vessels_found:
                prompt += "\n    - None"

    return prompt