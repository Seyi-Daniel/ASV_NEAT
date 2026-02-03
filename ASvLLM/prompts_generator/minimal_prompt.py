import math


def calculate_relative_bearing(own_ship, target_ship):
    """Calculates bearing of target relative to own ship's heading (0=ahead, 90=starboard/CW)."""
    dx = target_ship.x - own_ship.x
    dy = target_ship.y - own_ship.y
    world_angle_rad = math.atan2(-dy, dx)
    own_heading_rad = (math.pi / 2 - own_ship.heading) % (2 * math.pi)
    relative_angle_rad = world_angle_rad - own_heading_rad
    relative_bearing_rad = (-relative_angle_rad) % (2 * math.pi)
    return math.degrees(relative_bearing_rad)


# --- UPDATED SIGNATURE ---
def generate_vessel_prompt(vessels_to_describe, all_context_vessels, pixels_per_km=1, previous_vessel_data_list=None,
                           previous_responses=None):
    """
    Generates a minimal prompt.
    Note: This prompt style does NOT use history, but accepts the arguments for compatibility.
    """
    prompt = ("""
        "Ship navigation according to COLREGs.\n"
        "For each vessel listed below, determine the situation, role, action\n"
        "Respond ONLY with a JSON array of objects {id, situation, role, action}, **exactly one entry per vessel**. Do not produce more than one element for any given id\n\n"
        " Do not include any additional text.\n"
        "Vessels Data:"
        """
              )

    for v in vessels_to_describe:
        speed_kmh = (v.speed / pixels_per_km * 3600) if pixels_per_km > 0 else 0
        prompt += f"\n- id: {id(v)}, position: ({v.x:.1f}, {v.y:.1f}) px, heading: {math.degrees(v.heading):.1f}°, speed: {speed_kmh:.1f} km/h"
        prompt += "\n  Other vessels:"

        other_vessels_found = False
        for o in all_context_vessels:
            if o is v: continue

            # Note: Minimal prompt logic is slightly different (threshold check),
            # but for consistency with the new system, we generally pass all context vessels provided.
            # If you want to keep the speed threshold, you can add it back here:
            # if (o.speed / pixels_per_km * 3600) < 0.1: continue

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