import math
from entity import Vessel, Francisco

# Define a list of colors to cycle through for the vessels
VESSEL_COLORS = [
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (0, 255, 0),    # Green
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
]

def head_on_scenario(entity_manager, screen_width, screen_height):
    """
    Clears current vessels and creates two vessels placed on opposite sides
    of the screen, assigning each vessel the other's starting position as its goal.
    """
    # Optional: Clear existing vessels
    entity_manager.vessels.clear()

    # Define positions based on the screen dimensions.
    # For a head-on scenario, one vessel on the left, one on the right,
    # both centered vertically.
    left_x = screen_width * 0.1
    right_x = screen_width * 0.9
    center_y = screen_height / 2

    # Create two vessels (or BigVessels if desired)
    vessel1 = Vessel(left_x, center_y, 1000, color=VESSEL_COLORS[0])
    vessel2 = Vessel(right_x, center_y, 1000, 25, color=VESSEL_COLORS[1])

    vessel1.heading = math.pi / 2
    vessel2.heading = -math.pi / 2

    # Set goals: vessel1 moves toward vessel2's starting position, and vice versa.
    vessel1.goal = (right_x, center_y)
    vessel2.goal = (left_x, center_y)

    # Add the vessels to the entity manager
    entity_manager.vessels.append(vessel1)
    entity_manager.vessels.append(vessel2)


def cross_over_scenario(entity_manager, screen_width, screen_height):
    """
        Clears any existing vessels and creates two vessels arranged so that:
          - Vessel 1 moves from the mid-point of the lower screen to the mid-point of the top.
          - Vessel 2 moves from the mid-point of the right side to the mid-point of the left.
        """
    # Clear any existing vessels from the entity manager
    entity_manager.vessels.clear()

    # Vessel 1: from lower center to top center.
    vessel1_start_x = screen_width / 2
    vessel1_start_y = screen_height * 0.9  # near the bottom (90% down the screen)
    vessel1_goal_x = screen_width / 2
    vessel1_goal_y = screen_height * 0.1  # near the top (10% down the screen)

    # Vessel 2: from right center to left center.
    vessel2_start_x = screen_width * 0.9
    vessel2_start_y = screen_height / 2
    vessel2_goal_x = screen_width * 0.1
    vessel2_goal_y = screen_height / 2

    # Create the vessels.
    vessel1 = Vessel(vessel1_start_x, vessel1_start_y, 1000, color=VESSEL_COLORS[0])
    vessel2 = Vessel(vessel2_start_x, vessel2_start_y, 1000, color=VESSEL_COLORS[1])

    vessel2.heading = -math.pi / 2

    # Assign the respective goals
    vessel1.goal = (vessel1_goal_x, vessel1_goal_y)
    vessel2.goal = (vessel2_goal_x, vessel2_goal_y)

    # Add the vessels to the entity manager
    entity_manager.vessels.append(vessel1)
    entity_manager.vessels.append(vessel2)


def over_taking_scenario(entity_manager, screen_width, screen_height):
    """
           Clears any existing vessels and creates two vessels arranged so that:
             - Vessel 1 moves from the mid-point of the lower screen to the mid-point of the top.
             - Vessel 2 moves from the mid-point of the lower screen but with higher speed to the top of the screen.
             We can adjust the goal location of vessel 2.
           """
    # Clear any existing vessels from the entity manager
    entity_manager.vessels.clear()

    # Vessel 1: from lower center to top center.
    vessel1_start_x = screen_width / 2
    vessel1_start_y = screen_height * 0.7  # near the bottom (80% down the screen)
    vessel1_goal_x = screen_width / 2
    vessel1_goal_y = screen_height * 0.2  # near the top (20% down the screen)

    # Vessel 2: from lower center to top center.
    vessel2_start_x = screen_width / 2
    vessel2_start_y = screen_height * 0.9  # near the bottom (90% down the screen)
    vessel2_goal_x = screen_width / 2
    vessel2_goal_y = screen_height * (-10.5) # near the top (10% down the screen)

    # Create the vessels.
    vessel1 = Vessel(vessel1_start_x, vessel1_start_y, 1000, color=VESSEL_COLORS[0])
    vessel2 = Francisco(vessel2_start_x, vessel2_start_y, 1000)

    # Assign the respective goals
    vessel1.goal = (vessel1_goal_x, vessel1_goal_y)
    vessel2.goal = (vessel2_goal_x, vessel2_goal_y)

    # Add the vessels to the entity manager
    entity_manager.vessels.append(vessel1)
    entity_manager.vessels.append(vessel2)


def multi_vessel_scenario(entity_manager, screen_width, screen_height):
    """
    Clears existing vessels and creates three vessels at left, right, and bottom.
    They will move:
      - Left vessel heads east toward the right vessel.
      - Right vessel heads west toward the left vessel.
      - Bottom vessel heads north toward the midpoint of the top.
    """
    entity_manager.vessels.clear()

    left_x = screen_width * 0.1
    mid_y = screen_height / 2
    right_x = screen_width * 0.9
    top_x = screen_width / 2
    top_y = screen_height * 0.2
    bottom_x = screen_width / 2
    bottom_y = screen_height * 0.9

    vessel_left = Vessel(left_x, mid_y, 1000, color=VESSEL_COLORS[0])
    vessel_right = Vessel(right_x, mid_y, 1000, color=VESSEL_COLORS[1])
    vessel_bottom = Vessel(bottom_x, bottom_y, 1000, color=VESSEL_COLORS[2])

    # Goals
    vessel_left.goal = (right_x, mid_y)
    vessel_right.goal = (left_x, mid_y)
    vessel_bottom.goal = (top_x, top_y)

    # Headings:
    # Left vessel faces east (toward right)
    vessel_left.heading = math.pi / 2
    # Right vessel faces west (toward left)
    vessel_right.heading = -math.pi / 2
    # Bottom vessel faces north (toward top)
    vessel_bottom.heading = 0.0

    entity_manager.vessels.extend([vessel_left, vessel_right, vessel_bottom])


def multi_vessel_scenario_2(entity_manager, screen_width, screen_height):
    """
        Clears existing vessels and creates three vessels at left, right, and bottom.
        They will move:
          - Left vessel heads east toward the right vessel.
          - Right vessel heads west toward the left vessel.
          - Bottom vessel heads north toward the midpoint of the top.
        """
    entity_manager.vessels.clear()

    left_x = screen_width * 0.1
    mid_y = screen_height /3
    right_x = screen_width * 0.9
    top_x = screen_width / 2
    top_y = screen_height * 0.2
    bottom_x = screen_width / 2
    bottom_y = screen_height * 0.9
    right_goal = screen_height * 0.9

    vessel_top = Vessel(top_x, top_y, 1000, color=VESSEL_COLORS[0])
    vessel_right = Vessel(right_x, mid_y, 1000, color=VESSEL_COLORS[1])
    vessel_bottom = Vessel(bottom_x, bottom_y, 1000, color=VESSEL_COLORS[2])

    # Goals
    vessel_top.goal = (bottom_x, bottom_y)
    vessel_right.goal = (left_x, right_goal)
    vessel_bottom.goal = (top_x, top_y)

    # Headings:
    # Left vessel faces east (toward right)
    vessel_top.heading = math.pi
    # Right vessel faces west (toward left)
    vessel_right.heading = -math.pi / 1.5
    # Bottom vessel faces north (toward top)
    vessel_bottom.heading = 0.0

    entity_manager.vessels.extend([vessel_top, vessel_right, vessel_bottom])


# NEW SCENARIO
def traffic_separation_scenario(entity_manager, screen_width, screen_height):
    """
    Creates a Traffic Separation Scheme (TSS) with two lanes of traffic
    and a single user vessel tasked with crossing them.
    Traffic is visible immediately at the start of the scenario.
    """
    entity_manager.vessels.clear()

    # 1. Define the Traffic Lane Properties
    top_lane_y = screen_height * 0.4
    bottom_lane_y = screen_height * 0.6
    num_vessels_per_lane = 5
    lane_vessel_spacing = screen_width / 2.5

    # 2. Create the Eastbound Lane (Bottom Lane, Left to Right)
    for i in range(num_vessels_per_lane):
        # --- CHANGE: Start the first vessel ON the screen, others staged behind it. ---
        start_x = 10 - (i * lane_vessel_spacing)
        goal_x = screen_width + 500
        vessel = Vessel(start_x, bottom_lane_y, 1000, color=VESSEL_COLORS[2])  # Green traffic
        vessel.goal = (goal_x, bottom_lane_y)
        vessel.heading = math.pi / 2
        vessel.speed = vessel.max_speed  # Start at full speed
        entity_manager.add_vessel(vessel)

    # 3. Create the Westbound Lane (Top Lane, Right to Left)
    for i in range(num_vessels_per_lane):
        # --- CHANGE: Start the first vessel ON the screen, others staged behind it. ---
        start_x = (screen_width - 10) + (i * lane_vessel_spacing)
        goal_x = -500
        vessel = Vessel(start_x, top_lane_y, 1000, color=VESSEL_COLORS[3])  # Orange traffic
        vessel.goal = (goal_x, top_lane_y)
        vessel.heading = -math.pi / 2
        vessel.speed = vessel.max_speed  # Start at full speed
        entity_manager.add_vessel(vessel)

    # 4. Create the Player's Crossing Vessel (Red)
    crossing_vessel_start_x = screen_width / 2
    crossing_vessel_start_y = screen_height * 0.95
    crossing_vessel_goal_x = screen_width / 2
    crossing_vessel_goal_y = screen_height * 0.05

    crossing_vessel = Francisco(crossing_vessel_start_x, crossing_vessel_start_y, 1000)
    crossing_vessel.goal = (crossing_vessel_goal_x, crossing_vessel_goal_y)
    crossing_vessel.heading = 0.0
    entity_manager.add_vessel(crossing_vessel)