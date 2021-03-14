from enum import Enum
import random
import carla
import numpy as np
from Experiment.Utils.imports import RoadOption
from Experiment.Utils.imports import detect_lane_obstacle
random.seed(0)

class ActorState(Enum):
    """
    Represents the possible states of a scenario agent
    """
    STARTING = -1
    STOPPING = 0
    GOING = 1

class ActorBehavior(object):

    def __init__(self,constants):
        self.state = ActorState.STARTING
        self.stop_count = 0
        self.stop_thresh = 90
        self.random = 5
        self.constants = constants

    def run_step(self,vehicle,ego,local_planner, world):

        def at_intersection():
            return (self.constants.ego_spawn_loc.location.x - 17 > vehicle.get_location().x
                    > self.constants.ego_spawn_loc.location.x - 44)

        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v1 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / 3.1415926

        # calculat angle between vehicle and ego
        ego_vehicle_vector = ego.get_transform().get_forward_vector()
        target_vehicle_vector = vehicle.get_transform().get_forward_vector()
        e_vec = (ego_vehicle_vector.x,ego_vehicle_vector.y,ego_vehicle_vector.z)
        t_vec = (target_vehicle_vector.x,target_vehicle_vector.y,target_vehicle_vector.z)
        angle = angle_between(e_vec,t_vec)

        if (detect_lane_obstacle(vehicle, world, extension_factor = 1.0, margin = 2) and
            abs(ego.get_velocity().x) > 1 and
            ego.get_location().y > self.constants.ego_spawn_loc.location.y + 1):

            control = local_planner.run_step(debug=False)
            control.throttle = 0.0
            control.brake = 0.85
            return control

        if at_intersection() and self.state == ActorState.STARTING:
            rand = random.randrange(0,10)
            if angle < 175 and ego.get_location().y < self.constants.ego_spawn_loc.location.y + 8:
                if rand < self.random:
                    self.state = ActorState.STOPPING
                else:
                    self.state = ActorState.GOING
            else:
                self.state = ActorState.GOING

        if self.state == ActorState.STOPPING:
            if ego.get_velocity().x < 0.01 and vehicle.get_velocity().x < 0.01:
                self.stop_count += 1
            if ego.get_location().y > self.constants.ego_spawn_loc.location.y + 8 or (self.stop_count >= self.stop_thresh):
                self.state = ActorState.GOING # go if ego vehicle passes intersection
            else:
                control = local_planner.run_step(debug=False)
                control.throttle = 0
                control.brake = 0.6
                return control

        if self.state == ActorState.GOING:
            local_planner.set_speed(25)
            control = local_planner.run_step(debug=False)
            return control

        if self.state == ActorState.STARTING:
            local_planner.set_speed(25)
            control = local_planner.run_step(debug=False)
            return control

class EgoState(Enum):
    pass

class EgoBehavior(object):

    def __init__(self, constants, behavior_type):
        self.passing = False
        self.constants = constants
        self.behavior_type = behavior_type

    def run_step(self,actor,ego,local_planner, world):
        if ego.get_location().x > self.constants.control_start:
            local_planner.set_speed(20)
            control = local_planner.run_step(debug = False)
            return control

        if self.behavior_type != 5 and ego.get_location().x > self.constants.ego_spawn_loc.location.x - 15.5:
            control = local_planner.run_step(debug = False)
            return control

        if self.passing:
            control = local_planner.run_step(debug = False)
            return control

        if self.behavior_type != 5:
            if ego.get_location().y < self.constants.ego_spawn_loc.location.y + 0.2: # nudge forward
                control = local_planner.run_step(debug = False)
                control.throttle = 0.2
                control.brake = 0
                return control

        if self.behavior_type == 1 or self.behavior_type == 2:
            local_planner.set_speed(20)
            control = local_planner.run_step(debug = False)
            return control

        if self.behavior_type == 3 or self.behavior_type == 4:
            if ((actor.get_velocity().x < 0.01 or actor.get_location().x > self.constants.ego_spawn_loc.location.x - 17) and
                (185 < ego.get_location().x < self.constants.ego_spawn_loc.location.x - 16)):
                local_planner.set_speed(20)
                control = local_planner.run_step(debug = False)
                self.passing = True
                return control
            else:
                control = local_planner.run_step(debug = False)
                control.throttle = 0
                control.brake = 1
                return control

        if self.behavior_type == 5:
            if actor.get_location().x > self.constants.ego_spawn_loc.location.x - 11:
                local_planner.set_speed(20)
                control = local_planner.run_step(debug = False)
                return control
            else:
                control = local_planner.run_step(debug = False)
                control.throttle = 0
                control.brake = 1
                control.steer = 0
                return control
