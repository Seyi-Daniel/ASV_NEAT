import math
import json


# --- Helper Functions (Unchanged) ---
def get_direction(heading_deg):
    if 337.5 <= heading_deg or heading_deg < 22.5: return "North"
    if 22.5 <= heading_deg < 67.5: return "Northeast"
    if 67.5 <= heading_deg < 112.5: return "East"
    if 112.5 <= heading_deg < 157.5: return "Southeast"
    if 157.5 <= heading_deg < 202.5: return "South"
    if 202.5 <= heading_deg < 247.5: return "Southwest"
    if 247.5 <= heading_deg < 292.5: return "West"
    if 292.5 <= heading_deg < 337.5: return "Northwest"
    return "Unknown"


def get_relative_position(rel_bearing_deg):
    rel_bearing_deg = rel_bearing_deg % 360
    if 355 < rel_bearing_deg or rel_bearing_deg <= 5: return "dead ahead"
    if 5 < rel_bearing_deg <= 112.5: return "on the starboard side (crossing zone)"  # Adjusted to match COLREGs light sector
    if 112.5 < rel_bearing_deg <= 185: return "on the starboard quarter (overtaking zone)"
    if 185 < rel_bearing_deg <= 247.5: return "on the port quarter (overtaking zone)"
    if 247.5 < rel_bearing_deg <= 355: return "on the port side (crossing zone)"
    return "at an unknown bearing"


def calculate_relative_bearing(own_ship, target_ship):
    dx = target_ship.x - own_ship.x;
    dy = target_ship.y - own_ship.y
    world_angle_rad = math.atan2(-dy, dx)
    own_heading_rad = (math.pi / 2 - own_ship.heading) % (2 * math.pi)
    relative_angle_rad = world_angle_rad - own_heading_rad
    relative_bearing_rad = (-relative_angle_rad) % (2 * math.pi)
    return math.degrees(relative_bearing_rad)


def calculate_cpa_details(v1, v2, pixels_per_km):
    p_rel_x = v1.x - v2.x;
    p_rel_y = v1.y - v2.y
    v1_vx, v1_vy = v1.speed * math.sin(v1.heading), -v1.speed * math.cos(v1.heading)
    v2_vx, v2_vy = v2.speed * math.sin(v2.heading), -v2.speed * math.cos(v2.heading)
    v_rel_x = v1_vx - v2_vx;
    v_rel_y = v1_vy - v2_vy
    v_rel_mag_sq = v_rel_x ** 2 + v_rel_y ** 2
    if v_rel_mag_sq == 0: return None, None
    dot_product = (p_rel_x * v_rel_x) + (p_rel_y * v_rel_y)
    tcpa = -dot_product / v_rel_mag_sq
    if tcpa < 0: return None, None
    future_x1 = v1.x + v1_vx * tcpa;
    future_y1 = v1.y + v1_vy * tcpa
    future_x2 = v2.x + v2_vx * tcpa;
    future_y2 = v2.y + v2_vy * tcpa
    dist_sq = (future_x1 - future_x2) ** 2 + (future_y1 - future_y2) ** 2
    dcpa_km = math.sqrt(dist_sq) / pixels_per_km if pixels_per_km > 0 else 0
    return tcpa, dcpa_km


def calculate_historical_relative_bearing(own_vessel_state, target_vessel_state):
    try:
        own_x = float(own_vessel_state['pos'][0]);
        own_y = float(own_vessel_state['pos'][1])
        target_x = float(target_vessel_state['pos'][0]);
        target_y = float(target_vessel_state['pos'][1])
        dx = target_x - own_x;
        dy = target_y - own_y
        world_angle_rad = math.atan2(-dy, dx)
        own_heading_rad_0_is_north = math.radians(float(own_vessel_state['heading_deg']))
        own_heading_rad_trig = (math.pi / 2 - own_heading_rad_0_is_north) % (2 * math.pi)
        relative_angle_rad = world_angle_rad - own_heading_rad_trig
        relative_bearing_rad = (-relative_angle_rad) % (2 * math.pi)
        return math.degrees(relative_bearing_rad)
    except Exception as e:
        print(f"Error calculating historical bearing: {e}");
        return 0.0


# --- MAIN PROMPT GENERATION FUNCTION ---
def generate_natural_language_prompt(vessels_to_describe, all_context_vessels, pixels_per_km=1,
                                     previous_vessel_data_list=None, previous_responses=None):
    prompt = ("""
"You are a ship navigation expert system following COLREGs.
Analyze the provided image and the **current situation summary** below.
Also consider the **recent history** of the situation and decisions made.

"""
              )

    # --- HISTORY SECTION (Unchanged) ---
    if previous_vessel_data_list and previous_responses and len(previous_vessel_data_list) == len(previous_responses):
        prompt += "**Recent History (Most recent first):**\n"
        for i in range(len(previous_vessel_data_list) - 1, -1, -1):
            history_step = len(previous_vessel_data_list) - i
            prompt += f"--- History Step T-{history_step} ---\n"
            historical_states = previous_vessel_data_list[i]
            prompt += f"Situation Summary (T-{history_step}):\n"
            historical_decision_ids = [d['id'] for d in previous_responses[i]]
            for v_state in historical_states:
                if v_state['id'] in historical_decision_ids:
                    heading_deg_float = float(v_state['heading_deg'])
                    direction = get_direction(heading_deg_float)
                    speed_kmh = v_state['speed_kmh']
                    prompt += f"  - Vessel (ID: {v_state['id']}) was heading {direction} ({heading_deg_float:.1f}°) at {speed_kmh} km/h.\n"
                    prompt += "    Nearby vessels:\n"
                    other_vessels_found = False
                    for o_state in historical_states:
                        if o_state['id'] == v_state['id']: continue
                        hist_rb = calculate_historical_relative_bearing(v_state, o_state)
                        hist_dist_km = math.hypot(float(v_state['pos'][0]) - float(o_state['pos'][0]),
                                                  float(v_state['pos'][1]) - float(o_state['pos'][1])) / pixels_per_km
                        relative_pos = get_relative_position(hist_rb)
                        prompt += f"      - Vessel (ID: {o_state['id']}) was {hist_dist_km:.3f} km away, approaching from {relative_pos} ({hist_rb:.1f}°).\n"
                        other_vessels_found = True
                    if not other_vessels_found: prompt += "      - None.\n"
            prompt += f"Decision Made (T-{history_step}):\n"
            for decision in previous_responses[i]:
                prompt += f"  - id: {decision.get('id', 'N/A')}, situation: {decision.get('situation', 'N/A')}, role: {decision.get('role', 'N/A')}, action: {decision.get('action', 'N/A')}\n"
            prompt += "\n"
        prompt += "---\n\n"
    # --- END HISTORY SECTION ---

    # --- REFINED ANALYSIS INSTRUCTIONS ---
    # These definitions now align perfectly with COLREGs sectors
    prompt += """
**Current Situation Analysis & Action Required:**

**Analysis Instructions (COLREGs Rules):**
1.  **Overtaking (Rule 13):** Occurs when a vessel approaches from **more than 22.5° abaft the beam** (starboard/port quarter). The overtaking vessel is *always* the **Give-way** vessel and must keep clear.
2.  **Head-on (Rule 14):** Occurs when two vessels are on reciprocal or nearly reciprocal courses, seeing each other **dead ahead** (approx 0° relative). Both are **Give-way** and must alter course to starboard.
3.  **Crossing (Rule 15):** Occurs when a vessel approaches from the **starboard side** (starboard bow/beam, 5° to 112.5°). The vessel with the other on its own starboard side is the **Give-way** vessel.
4.  **Roles:** - **Give-way:** Must take early and substantial action to keep well clear (usually 'Alter course to starboard').
    - **Stand-on:** Must maintain course and speed, unless collision is imminent.

**Current Situation Summary:**
"""

    # --- Build Natural Language Description for CURRENT time ---
    for v in vessels_to_describe:
        speed_kmh = (v.speed / pixels_per_km * 3600) if pixels_per_km > 0 else 0
        heading_deg = math.degrees(v.heading)
        direction = get_direction(heading_deg)

        prompt += f"Vessel (ID: {id(v)}) is heading {direction} ({heading_deg:.1f}°) at {speed_kmh:.1f} km/h.\n"
        prompt += "Nearby vessels:\n"

        other_vessels_found = False
        threats = []
        for o in all_context_vessels:
            if o is v: continue

            dx, dy = o.x - v.x, o.y - v.y
            dist_km = math.hypot(dx, dy) / pixels_per_km if pixels_per_km > 0 else 0
            rb = calculate_relative_bearing(v, o)
            sp = (o.speed / pixels_per_km * 3600) if pixels_per_km > 0 else 0

            relative_pos = get_relative_position(rb)
            tcpa, dcpa_km = calculate_cpa_details(v, o, pixels_per_km)

            motion_desc = ""
            cpa_info = ""
            if tcpa is not None and dcpa_km is not None:
                # Threat is in the future
                motion_desc = f"and is APPROACHING from {relative_pos} ({rb:.1f}°)."
                if tcpa < 120:
                    cpa_info = f"    WARNING: Risk of collision detected. TCPA: {tcpa:.1f}s, DCPA: {dcpa_km:.3f}km.\n"
            else:
                # No future collision risk
                motion_desc = f"at {relative_pos} ({rb:.1f}°) and is MOVING AWAY or on a non-conflicting path."

            threat_str = f"  - Vessel (ID: {id(o)}) is {dist_km:.3f} km away, {motion_desc}\n"
            threat_str += cpa_info

            threats.append((tcpa if tcpa is not None else 9999, threat_str))
            other_vessels_found = True

        threats.sort()
        for _, threat_str in threats:
            prompt += threat_str

        if not other_vessels_found:
            prompt += "  - None.\n"

    # --- Output format MUST remain JSON ---
    prompt += """
---
**Task:**
Based on the image and the summary above, provide a JSON response for **each 'Vessel (ID: ...)'** listed (not the 'Nearby vessels').

**Output Format Requirements:**
JSON object must have keys: "id", "situation", "role", "action".
1.  `"id"`: Integer ID.
2.  `"situation"`: Brief description (e.g., "Overtaking", "Crossing threat on starboard bow").
3.  `"role"`: Role relative to immediate threat (e.g., "Give-way", "Stand-on").
4.  `"action"`: ONE of the exact phrases below:
    - 'Alter course to starboard'
    - 'Alter course to port'
    - 'Reduce speed'
    - 'Pass astern'
    - 'Accelerate'
    - 'Maintain course and speed'
"""
    return prompt