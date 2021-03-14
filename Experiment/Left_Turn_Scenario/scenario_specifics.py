import carla
import random
from Experiment.Utils.imports import RoadOption
from Experiment.Utils.imports import retrieve_options
random.seed(0)

TOWN = ['Town04','Town04','Town04','Town04']
ACTOR_SPAWN_LOC = [carla.Transform(carla.Location(160,-170,3),carla.Rotation(0,0,0)),
                carla.Transform(carla.Location(217,-169.5,3),carla.Rotation(0,0,0)),
                carla.Transform(carla.Location(272,-169.0,3),carla.Rotation(0,0,0)),
                carla.Transform(carla.Location(217,-246.5,3),carla.Rotation(0,0,0))]

EGO_SPAWN_LOC = [carla.Transform(carla.Location(229,-173,3),carla.Rotation(0,180,0)),
                carla.Transform(carla.Location(286,-172.5,3),carla.Rotation(0,180,0)),
                carla.Transform(carla.Location(341,-172.0,3),carla.Rotation(0,180,0)),
                carla.Transform(carla.Location(286,-249.5,3),carla.Rotation(0,180,0))]
CONTROL_START = [219,274.5,329,276]
GOAL = [-160,-160,-160,-237]

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
        return ego.get_location().x <= self.control_start

    def reached_goal(self,ego):
        return ego.get_location().y >= self.goal

    def get_starting_plans(self,world):
        ego_plan = []
        target_waypoint = world.get_map().get_waypoint(self.ego_spawn_loc.location)
        loc = carla.Location(target_waypoint.transform.location.x - 9,target_waypoint.transform.location.y, 0)
        target_waypoint = world.get_map().get_waypoint(loc)
        wp_choice = target_waypoint.next(1.0)
        while len(ego_plan) < 150:
            if len(wp_choice) == 1:
                    target_waypoint = wp_choice[0]
                    road_option = RoadOption.LANEFOLLOW
            else:
                target_waypoint = max(wp_choice,key = lambda w:w.transform.location.y)
                loc = carla.Location(target_waypoint.transform.location.x,target_waypoint.transform.location.y + 0.1, 0)
                target_waypoint = world.get_map().get_waypoint(loc)
                road_option = RoadOption.LEFT

                #road_options_list = retrieve_options(wp_choice, ego_plan[-1][0])
                #target_waypoint = max(wp_choice,key = lambda w:w.transform.location.y)
                #road_option = road_options_list[wp_choice.index(target_waypoint)]
            ego_plan.append((target_waypoint,road_option))
            wp_choice = target_waypoint.next(1.0)

        actor_plan = []
        target_waypoint = world.get_map().get_waypoint(self.actor_spawn_loc.location)
        wp_choice = target_waypoint.next(1.0)
        while len(actor_plan) < 200:
            if len(wp_choice) == 1:
                    target_waypoint = wp_choice[0]
            else:
                loc = carla.Location(target_waypoint.transform.location.x + 4,target_waypoint.transform.location.y, 0)
                target_waypoint = world.get_map().get_waypoint(loc)
            actor_plan.append((target_waypoint,RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(1.0)

        return ego_plan,actor_plan
