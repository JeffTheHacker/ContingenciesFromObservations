from enum import Enum
import random
import carla
from Experiment.Utils.imports import RoadOption
random.seed(0)

class ActorState(Enum):
    """
    Represents the possible states of a scenario agent
    """
    STARTING = -1
    BRAKE = 0
    AGGRES = 1
    YIELD = 2

class ActorBehavior(object):

    def __init__(self, constants):
        self.state = ActorState.STARTING
        self.random = 5
        self.stop_thresh = 90
        self.stop_count = 0
        self.overconfident_ego = False
        self.constants = constants

    def run_step(self,actor,ego,local_planner, world):

        def same_lane():
            return actor.get_location().x + 1 >= ego.get_location().x >= actor.get_location().x - 1

        def constant_speed(speed,throttle = None,brake = None):
            local_planner.set_speed(speed)
            control = local_planner.run_step(debug = False)
            if throttle:
                control.throttle = throttle
            if brake:
                control.brake = brake
            return control

        if ego.get_velocity().y > 10.5 and ego.get_location().y < actor.get_location().y - 2:
            self.overconfident_ego = True

        if self.state == ActorState.STARTING:
            if self.overconfident_ego and ego.get_location().y > actor.get_location().y + 8:
                rand = random.randrange(0,10)
                if rand < self.random: # yield
                    self.state = ActorState.BRAKE
                else:
                    self.state = ActorState.AGGRES
            if not self.overconfident_ego and ego.get_location().y > actor.get_location().y - 2:
                rand = random.randrange(0,10)
                if rand < self.random: # yield
                    self.state = ActorState.YIELD
                else:
                    self.state = ActorState.AGGRES
            return constant_speed(20)

        if same_lane() or self.state == ActorState.AGGRES or self.stop_count > self.stop_thresh:
            return constant_speed(20)

        if self.state == ActorState.BRAKE:
            return constant_speed(0,throttle = 0, brake = 0.5)

        if self.state == ActorState.YIELD:
            self.stop_count += 1
            return constant_speed(0,throttle = 0)

class EgoState(Enum):
    """
    Represents the possible states of a scenario agent
    """
    STARTING = -1
    TURNING = 0
    SIDE_BY_SIDE = 1
    TURNING_BACK = 2
    SLOWING = 3
    GOING = 4
    STAYING_BEHIND = 5

class EgoBehavior(object):

    def __init__(self, constants, behavior_type):
        self.state = EgoState.STARTING
        self.decelerate_count = 0
        self.ego_wait = 0
        self.constants = constants
        self.behavior_type = behavior_type

    def run_step(self,actor,ego,local_planner, world):

        def change_lane(direction):
            if direction == 0: # change to left
                target_waypoint = world.get_map().get_waypoint(ego.get_location()).get_left_lane().next(10)[0]
                plan = [(target_waypoint,RoadOption.CHANGELANELEFT)]
            else:
                right_loc = carla.Location(ego.get_location().x - 3,ego.get_location().y + 10,ego.get_location().z)
                target_waypoint = world.get_map().get_waypoint(right_loc)
                plan = [(target_waypoint,RoadOption.CHANGELANERIGHT)]
            for i in range(50):
                loc = carla.Location(target_waypoint.transform.location.x,target_waypoint.transform.location.y + 4, 0)
                target_waypoint = world.get_map().get_waypoint(loc)
                plan.append((target_waypoint,RoadOption.LANEFOLLOW))
            local_planner.set_global_plan(plan)
            local_planner._waypoint_buffer.clear()

        def constant_speed(speed,throttle = None,brake = None):
            local_planner.set_speed(speed)
            control = local_planner.run_step(debug = False)
            if throttle:
                control.throttle = throttle
            if brake:
                control.brake = brake
            return control

        if self.behavior_type == 5:
            return constant_speed(20)

        if self.state == EgoState.STAYING_BEHIND:
            return constant_speed(20)

        if self.state == EgoState.GOING:
            self.decelerate_count += 1
            if self.decelerate_count < 60:
                return constant_speed(15)
            else:
                return constant_speed(30)

        if self.state == EgoState.SLOWING:
            if ego.get_location().y < actor.get_location().y - 3:
                change_lane(1)
                self.state = EgoState.STAYING_BEHIND
            return constant_speed(0, throttle = None, brake = 0.5)

        if self.state == EgoState.TURNING_BACK:
            if actor.get_location().x + 1.5 >= ego.get_location().x:
                self.state = EgoState.GOING
            if self.behavior_type == 4:
                return constant_speed(0)
            else:
                return constant_speed(0, throttle = None, brake = 1)

        if self.state == EgoState.SIDE_BY_SIDE:
            if self.behavior_type == 1 or self.behavior_type == 2:
                if ego.get_location().y > actor.get_location().y + 5:
                    change_lane(1)
                    self.state = EgoState.TURNING_BACK
            elif ego.get_location().y > actor.get_location().y + 3:
                change_lane(1)
                self.state = EgoState.TURNING_BACK
            elif self.ego_wait >= 50:
                self.state = EgoState.SLOWING
            self.ego_wait += 1
            if self.behavior_type == 1 or self.behavior_type == 2:
                return constant_speed(30, throttle = 1, brake = 0)
            else:
                return constant_speed(30)

        if self.state == EgoState.TURNING:
            if ego.get_location().y > actor.get_location().y - 2:
                self.state = EgoState.SIDE_BY_SIDE
            if self.behavior_type == 1 or self.behavior_type == 2:
                return constant_speed(30, throttle = 1, brake = 0)
            else:
                return constant_speed(30)

        if self.state == EgoState.STARTING:
            if ego.get_location().y > self.constants.control_start and ego.get_location().y != 0:
                self.state = EgoState.TURNING
                change_lane(0)
            return constant_speed(20)
