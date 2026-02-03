import json
import enum


# Maneuver enum definition
class Maneuver(enum.IntEnum):
    MAINTAIN_COURSE_SPEED = 0
    ALTER_COURSE_STARBOARD = 1
    ALTER_COURSE_PORT = 2
    REDUCE_SPEED = 3
    PASS_ASTERN = 4
    ACCELERATE = 5


def parse_llm_response_for_all(response_json_string: str):
    """
    Parses a JSON array response from LLM and maps actions to Maneuver enum.
    This version is more robust and uses keywords to handle responses from
    various LLMs like Claude, GPT, and DeepSeek.
    """
    try:
        data = json.loads(response_json_string)
        if not isinstance(data, list):
            print("Warning: Expected a JSON array but got:", data)
            return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON: {response_json_string}")
        return []

    results = []
    for entry in data:
        raw_id = entry.get("id")
        try:
            vid = int(raw_id)
        except (ValueError, TypeError):
            print(f"Warning: invalid id '{raw_id}', skipping entry")
            continue

        action_str = (entry.get("action") or "").lower()

        # --- NEW, MORE ROBUST MAPPING LOGIC ---
        # We check for keywords instead of exact phrases.
        # The order is important: check for the most specific terms first.

        # --- UPDATED MAPPING LOGIC ---
        if "astern" in action_str:
            m = Maneuver.PASS_ASTERN
        elif "starboard" in action_str:
            m = Maneuver.ALTER_COURSE_STARBOARD
        elif "port" in action_str:
            m = Maneuver.ALTER_COURSE_PORT
        elif "reduce" in action_str or "slow" in action_str:
            m = Maneuver.REDUCE_SPEED
        # --- ADD ACCELERATE CHECK ---
        elif "accelerate" in action_str or "increase speed" in action_str:
            m = Maneuver.ACCELERATE
        elif "keep clear" in action_str or "avoid" in action_str:  # Keep fallback
            m = Maneuver.ALTER_COURSE_STARBOARD
        else:
            m = Maneuver.MAINTAIN_COURSE_SPEED

        results.append({
            "id": vid,
            "action": entry.get("action"),
            "maneuver": m,
            "situation": entry.get("situation"),
            "role": entry.get("role")
        })
    return results
