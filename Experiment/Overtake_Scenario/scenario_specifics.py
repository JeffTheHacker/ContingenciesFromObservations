import carla
import random
from Experiment.Utils.imports import RoadOption
random.seed(0)

TOWN = ['Town03','Town05','Town05','Town05']
ACTOR_SPAWN_LOC = [carla.Transform(carla.Location(-88.5,-63,3),carla.Rotation(0,90,0)),
                   carla.Transform(carla.Location(25.3,-67.5,3),carla.Rotation(0,90,0)),
                   carla.Transform(carla.Location(24.4,20,3),carla.Rotation(0,90,0)),
                   carla.Transform(carla.Location(-54.6,-67.5,3),carla.Rotation(0,90,0))]

EGO_SPAWN_LOC = [carla.Transform(carla.Location(-88.5,-70,3),carla.Rotation(0,90,0)),
                 carla.Transform(carla.Location(25.3,-74,3),carla.Rotation(0,90,0)),
                 carla.Transform(carla.Location(24.4,13,3),carla.Rotation(0,90,0)),
                 carla.Transform(carla.Location(-54.6,-74,3),carla.Rotation(0,90,0))]
CONTROL_START = [-50,-54,33,-54]
GOAL = [20,16,103,16]

class ScenarioSpecifics(object):

    def __init__(self):
        self.resample(0)

    def resample(self,choice):
        self.choice = choice
        self.town = TOWN[self.choice]
        self.actor_spawn_loc = ACTOR_SPAWN_LOC[self.choice]
        self.ego_spawn_loc = EGO_SPAWN_LOC[self.choice]
        self.control_start = CONTROL_START[self.choice]
        self.goal = GOAL[self.choice]

    def past_control_start(self,ego):
        return ego.get_location().y >= self.control_start

    def reached_goal(self,ego):
        return ego.get_location().y >= self.goal

    def get_starting_plans(self,world):
        ego_plan = []
        target_waypoint = world.get_map().get_waypoint(self.ego_spawn_loc.location)
        while len(ego_plan) < 50:
            loc = carla.Location(target_waypoint.transform.location.x,target_waypoint.transform.location.y + 4, 0)
            target_waypoint = world.get_map().get_waypoint(loc)
            ego_plan.append((target_waypoint,RoadOption.LANEFOLLOW))

        actor_plan = []
        target_waypoint = world.get_map().get_waypoint(self.actor_spawn_loc.location)
        while len(actor_plan) < 50:
            loc = carla.Location(target_waypoint.transform.location.x,target_waypoint.transform.location.y + 4, 0)
            target_waypoint = world.get_map().get_waypoint(loc)
            actor_plan.append((target_waypoint,RoadOption.LANEFOLLOW))

        return ego_plan, actor_plan
