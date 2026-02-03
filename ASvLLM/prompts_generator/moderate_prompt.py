import math


def calculate_relative_bearing(own_ship, target_ship):
    dx = target_ship.x - own_ship.x
    dy = target_ship.y - own_ship.y
    world_angle = math.atan2(-dy, dx)
    own_heading = (math.pi / 2 - own_ship.heading) % (2 * math.pi)
    rel = world_angle - own_heading
    return math.degrees((-rel) % (2 * math.pi))


# --- UPDATED SIGNATURE ---
def generate_vessel_prompt(vessels_to_describe, all_context_vessels, pixels_per_km=1, previous_vessel_data_list=None,
                           previous_responses=None):
    prompt = ("""
        "Ship navigation according to COLREGs.\n"
        "For each vessel listed below, determine the situation, role, action\n"
        "Respond ONLY with a JSON array of objects {id, situation, role, action}, **exactly one entry per vessel**. Do not produce more than one element for any given id\n\n"
        " Do not include any additional text.\n\n"

    Types:
      - situation: "Head-on", "Crossing", or "Overtaking"
      - role: "Give-way" or "Stand-on"
      - action: "Alter course to starboard", "Alter course to port", "Reduce speed", or "Maintain course and speed"

    Strict rules:
      1. Head-on:  both vessels → Give-way, action → "Alter course to starboard"
      2. Crossing: vessel seeing the other on its starboard side → Give-way; other → Stand-on
      3. Overtaking: faster vessel → Give-way; slower → Stand-on

    Vessels Data:
    """
              )
    for v in vessels_to_describe:
        speed_kmh = (v.speed / pixels_per_km * 3600) if pixels_per_km else 0
        prompt += f"- id: {id(v)}, pos: ({v.x:.1f},{v.y:.1f}) px, heading: {math.degrees(v.heading):.1f}°, speed: {speed_kmh:.1f} km/h\n"
        prompt += "  Other vessels:\n"

        other_vessels_found = False
        for o in all_context_vessels:
            if o is v: continue

            dx, dy = o.x - v.x, o.y - v.y
            dist = math.hypot(dx, dy) / pixels_per_km if pixels_per_km else 0
            rb = calculate_relative_bearing(v, o)
            sp = (o.speed / pixels_per_km * 3600) if pixels_per_km else 0
            prompt += (
                f"    - id: {id(o)}, pos: ({o.x:.1f},{o.y:.1f}) px, "
                f"heading: {math.degrees(o.heading):.1f}°, speed: {sp:.1f} km/h, "
                f"dist: {dist:.3f} km, relBearing: {rb:.1f}°\n"
            )
            other_vessels_found = True

        if not other_vessels_found:
            prompt += "    - None\n"

    return prompt