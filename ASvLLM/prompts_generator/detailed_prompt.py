import math


def calculate_relative_bearing(own_ship, target_ship):
    dx = target_ship.x - own_ship.x
    dy = target_ship.y - own_ship.y
    world_angle = math.atan2(-dy, dx)
    own_heading = (math.pi / 2 - own_ship.heading) % (2 * math.pi)
    rel = world_angle - own_heading
    return math.degrees((-rel) % (2 * math.pi))


def compute_tcpa_dcpa(own_ship, other_ship, pixels_per_km):
    dx = (other_ship.x - own_ship.x) / pixels_per_km
    dy = (other_ship.y - own_ship.y) / pixels_per_km
    rvx = (other_ship.desired_velocity[0] - own_ship.desired_velocity[0]) / pixels_per_km
    rvy = (other_ship.desired_velocity[1] - own_ship.desired_velocity[1]) / pixels_per_km
    rv2 = rvx ** 2 + rvy ** 2
    if rv2 == 0:
        return float('inf'), math.hypot(dx, dy)
    tcpa = - (dx * rvx + dy * rvy) / rv2
    cx = dx + rvx * tcpa
    cy = dy + rvy * tcpa
    dcpa = math.hypot(cx, cy)
    return tcpa, dcpa


# --- UPDATED SIGNATURE ---
def generate_vessel_prompt(vessels_to_describe, all_context_vessels, pixels_per_km=1, previous_vessel_data_list=None,
                           previous_responses=None):
    prompt = (
        """
        "Ship navigation according to COLREGs.\n"
        "For each vessel listed below, determine the situation, role, action\n"

        Types:
      - situation: "Head-on", "Crossing", or "Overtaking"
      - role: "Give-way" or "Stand-on"
      - action: "Alter course to starboard", "Alter course to port", "Reduce speed", or "Maintain course and speed"

        "Strict rules for labeling encounters:\n"
        "1) Head-on: If vessels not moving to same direction and vessels on reciprocal or near-reciprocal courses (bearing diff ≈ 180°) → both \"Give-way\" & action \"Alter course to starboard\".\n"
        "2) Crossing: If vessel B is on vessel A’s starboard side (relative bearing 0°<θ<112.5°) → A is \"Give-way\", B is \"Stand-on\".\n"
        "3) Overtaking: if vessels are on the same course (heading difference <22.5°) and speed difference >5 km/h and the other vessel’s relBearing is >112.5° and <247.5°. Faster vessel: Give-way, Alter course to starboard; slower vessel: Stand-on.\n"

        "Respond ONLY with a JSON array of objects {id, situation, role, action}, **exactly one entry per vessel**. Do not produce more than one element for any given id, and do not include any extra text.\n\n"
        "Vessels Data:\n"
         """
    )

    for v in vessels_to_describe:
        speed_kmh = (v.speed / pixels_per_km * 3600)
        prompt += f"- id: {id(v)}, pos: ({v.x:.1f},{v.y:.1f}), heading: {math.degrees(v.heading):.1f}°, speed: {speed_kmh:.1f} km/h\n"
        prompt += "  Other vessels:\n"

        other_vessels_found = False
        for o in all_context_vessels:
            if o is v: continue

            # Note: Using context vessels passed from EntityManager
            dist_km = math.hypot(o.x - v.x, o.y - v.y) / pixels_per_km
            rb = calculate_relative_bearing(v, o)
            other_speed = (o.speed / pixels_per_km * 3600)

            tcpa, dcpa = compute_tcpa_dcpa(v, o, pixels_per_km)
            prompt += (
                f"    - id: {id(o)}, pos: ({o.x:.1f},{o.y:.1f}), heading: {math.degrees(o.heading):.1f}°, "
                f"speed: {other_speed:.1f} km/h, "
                f"dist: {dist_km:.3f} km, relBearing: {rb:.1f}°, "
                f"TCPA: {tcpa:.1f}s, DCPA: {dcpa:.3f} km\n"
            )
            other_vessels_found = True

        if not other_vessels_found:
            prompt += "    - None\n"

    return prompt