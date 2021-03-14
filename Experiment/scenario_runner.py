import sys
sys.path.append('../')
from Utils.imports import *
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--enable-recording', dest='recording',action='store_true', help='Enable recording of Carla experiments.')
parser.add_argument('--enable-sampling', dest='sampling',action='store_true', help='Plot sampled trajectories at every frame.')
parser.add_argument('--enable-evaluation', dest='evaluation',action='store_true', help='Evaluate likelihood or each expert behavior type.')
parser.add_argument('--enable-collecting', dest='collecting',action='store_true', help='Collect data.')
parser.add_argument('-cp', '--checkpoint_path', action='store', dest='checkpoint_path', type=str, help='Absolute path to model checkpoint, example: ...SocialConvRNN_/esp-model-100000')
parser.add_argument('-mp', '--model_path', action='store', dest='model_path', type=str, help='Absolute path to model folder, example: ...SocialConvRNN_')
parser.add_argument('--enable-inference', dest='inference',action='store_true', help='Run planning and plot trajectories in Carla.')
parser.add_argument('--enable-control', dest='control',action='store_true', help='Control the ego vehicle with planned trajectories.')
parser.add_argument('-r', '--replan', action='store', type=int, dest='replan', help='Integer > 0. Sets replanning rate. Smaller -> more frequent replanning.')
parser.add_argument('-t', '--planner_type', action='store', type=int, dest='planner_type', help='Sets the type of planner. 0: contingent, 1: overconfident, 2: underconfident.')
parser.add_argument('-s', '--scenario', action='store', type=int, dest='scenario', help='Sets the scenario. 0: left turn, 1: overtake, 2: right turn')
parser.add_argument('-l', '--location', action='store', type=int, dest='location', help='Sets location of the scenario, an integer from 0 to 3.')
args = parser.parse_args()

# configurations
RECORDING = args.recording # if true, will capture images and turn images into video upon exit.
SAMPLING = args.sampling # if true, will run inference at every frame and draw sampled trajectories on screen
EVALUATION = args.evaluation # if true, will evaluate likelihood of each of the expert behavior types
COLLECTING = args.collecting # if true, will collect data of episode
CHECKPOINT_PATH = args.checkpoint_path
MODEL_PATH = args.model_path
INFERENCE = args.inference # if true, will run planning with goal likelihood
CONTROL = args.control # if true, will use our own control, otherwise uses autopilot
REPLAN = args.replan # this means we want to use REPLAN number of waypoints before we replan
PLANNER_TYPE = args.planner_type # 0: contingent, 1: overconfident, 2: underconfident
SCENARIO = args.scenario # 0: left turn, 1: overtake, 2: right turn
LOCATION = args.location # 0-3, representing locations.

if SCENARIO == 0:
    from Experiment.Left_Turn_Scenario.behaviors import *
    from Experiment.Left_Turn_Scenario.scenario_specifics import *
    from Experiment.Left_Turn_Scenario.goal_likelihood import *
if SCENARIO == 1:
    from Experiment.Overtake_Scenario.behaviors import *
    from Experiment.Overtake_Scenario.scenario_specifics import *
    from Experiment.Overtake_Scenario.goal_likelihood import *
if SCENARIO == 2:
    from Experiment.Right_Turn_Scenario.behaviors import *
    from Experiment.Right_Turn_Scenario.scenario_specifics import *
    from Experiment.Right_Turn_Scenario.goal_likelihood import *

class ScenarioRunner(object):

    def start(self):
        # create world
        self.spawn_world('Town03')

        if SAMPLING or INFERENCE or CONTROL or EVALUATION:
            # start loading the model for running inference
            self.sess = tfutil.create_session(allow_growth=True, per_process_gpu_memory_fraction=0.5)
            log.info("Loading the model...")
            ckpt, graph, self.tensor_collections = tfutil.load_annotated_model(MODEL_PATH, self.sess, CHECKPOINT_PATH)
            self.inference = interface.ESPInference(self.tensor_collections)

        # spawn ego and actor
        self.ego_bp = self.world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        self.actor_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")

        # set up lidar sensor
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('sensor_tick', str(1/30))
        self.lidar_bp.set_attribute('range',str(40))
        self.lidar_bp.set_attribute('points_per_second',str(270000))
        self.lidar_bp.set_attribute('rotation_frequency',str(240))
        self.lidar_bp.set_attribute('channels',str(64))

        # set up camera
        self.cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.cam_bp.set_attribute("image_size_x",str(1920))
        self.cam_bp.set_attribute("image_size_y",str(1080))
        self.cam_bp.set_attribute("fov",str(105))
        self.cam_bp.set_attribute("sensor_tick",str(1/30))

        # set up collision sensor
        self.col_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # for destroying purposes
        self.actor_list = []

        # behavior types
        self.expert_behaviors = [(1,0),(2,10),(3,0),(4,10),(5,0)]
        self.current_behavior = -1

        self.collector = DataCollector()

        if PLANNER_TYPE == 2:
            # key: 0,1,2,3,4 representing the 5 scenarios
            # value: 2 * n * 2 with n representing scenario length in terms of frames
            self.all_trajectories = {}

        self.reset()

        # run this indefinitely
        while True:

            if (PLANNER_TYPE == 2 and self.episode_num < 5):
                self.check_episode_end()
                self.collect_data()
                self.handle_actor()
                self.handle_ego()
                self.world.tick()
                continue

            self.check_episode_end()
            self.collect_data()
            self.handle_actor()
            if EVALUATION and self.constants.past_control_start(self.ego):
                self.evaluate_mode_prob()
            if not CONTROL or len(self.past_ego_buffer) != 45 or not self.constants.past_control_start(self.ego):
                self.handle_ego()
            if SAMPLING and self.constants.past_control_start(self.ego):
                self.generate_sample()
            if INFERENCE and self.constants.past_control_start(self.ego):
                self.plan()
            self.world.tick()

    def spawn_world(self, name):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.world = self.client.load_world(name)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1/30 # 30 frames per second
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        weather = carla.WeatherParameters(cloudiness = 0, precipitation = 0, sun_altitude_angle = 90)
        self.world.set_weather(weather)
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.freeze(True)

    def check_episode_end(self):
        if self.constants.reached_goal(self.ego):
            self.count += 1
        if (self.count > 250 or self.collided):
            self.reset()

    def reset(self):

        # for checking episode end purposes
        self.count = 0

        # episode variables
        if hasattr(self,'episode_num'):
            self.episode_num += 1
        else:
            self.episode_num = 0

        # for constants
        self.constants = ScenarioSpecifics()
        self.constants.resample(choice=LOCATION)
        if (PLANNER_TYPE == 2 and self.episode_num < 5):
            self.constants.resample(choice=0)

        # create planner
        if INFERENCE or CONTROL:
            if PLANNER_TYPE == 0:
                self.planner = ContingentPlanner(self.inference, CostedGoalLikelihood(self.inference, self.constants),self.sess)
            elif PLANNER_TYPE == 1:
                self.planner = OverconfidentPlanner(self.inference, CostedGoalLikelihood(self.inference, self.constants),self.sess)
            elif PLANNER_TYPE == 2:
                self.planner = UnderconfidentPlanner(self.inference,CostedGoalLikelihood(self.inference,self.constants),self.sess,self.all_trajectories)

        # handle world creation
        if self.constants.town != self.world.get_map().name:
            self.spawn_world(self.constants.town)
        else:
            # remove all actors
            if hasattr(self,'ego') or hasattr(self,'actor'):
                self.lidar_sensor.destroy()
                self.record_cam.destroy()
                self.collison_sensor.destroy()
                self.ego.destroy()
                self.actor.destroy()

        # get the starting plans
        self.ego_plan, self.actor_plan = self.constants.get_starting_plans(self.world)

        # data collection variables
        self.past_ego_buffer = [] # stores past positions, used for inference
        self.past_actor_buffer = []
        self.lidar_data = None # stores lidar data


        # control variables
        self.future_plan = []

        # spawn ego
        self.ego = self.world.spawn_actor(self.ego_bp, self.constants.ego_spawn_loc)

        lidar_transform = carla.Transform(carla.Location(0,0,3),carla.Rotation(0,0,0))
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, lidar_transform, attach_to=self.ego)
        self.lidar_sensor.listen(self.lidar_callback)

        cam_transform = carla.Transform(carla.Location(6,0,50),carla.Rotation(-90,0,0))
        self.record_cam = self.world.spawn_actor(self.cam_bp,cam_transform, attach_to = self.ego)
        if RECORDING: # save images if we are recording
            self.record_cam.listen(self.camera_callback)

        self.collided = False
        self.collison_sensor = self.world.spawn_actor(self.col_bp,carla.Transform(carla.Location(0,0,0),carla.Rotation(0,0,0)),attach_to = self.ego)
        self.collison_sensor.listen(self.collision_callback)

        self.ego_local_planner = LocalPlanner(self.ego)
        self.ego_local_planner.set_global_plan(self.ego_plan)

        self.actor = self.world.spawn_actor(self.actor_bp,self.constants.actor_spawn_loc)
        self.actor_local_planner = LocalPlanner(self.actor)
        self.actor_local_planner.set_global_plan(self.actor_plan)

        # for destroying purposes
        self.actor_list = [self.ego,self.actor,self.lidar_sensor,self.record_cam,self.collison_sensor]

        if not CONTROL or (PLANNER_TYPE == 2 and self.episode_num < 5):

            # behavior type:
            # 1: ego enter, actor no yield, ego no yield, actor hard brake
            # 2: ego enter, actor no yield, ego no yield, actor yield
            # 3: ego enter, actor no yield, ego yield, actor passes
            # 4: ego enter, actor yields, ego doesn't yield
            # 5: ego doesn't enter, actor passes

            self.current_behavior = (self.current_behavior + 1) % len(self.expert_behaviors)
            self.behavior_type = self.expert_behaviors[self.current_behavior][0]

            print("Currently at episode %d" % (self.episode_num))
            print("Behavior type for this episode: %d" % self.behavior_type)

            self.actor_behavior = ActorBehavior(self.constants)
            self.actor_behavior.random = self.expert_behaviors[self.current_behavior][1]

            self.ego_behavior = EgoBehavior(self.constants, self.behavior_type)

            if PLANNER_TYPE == 2:
                self.all_trajectories[self.episode_num] = [[],[]]

        else:

            if PLANNER_TYPE == 2:
                for k in self.all_trajectories.keys():
                    self.all_trajectories[k] = np.array(self.all_trajectories[k])

            self.behavior_type = 1
            self.actor_behavior = ActorBehavior(self.constants)
            self.actor_behavior.random = 5
            self.ego_behavior = EgoBehavior(self.constants, self.behavior_type)
            print("Currently at episode %d" % (self.episode_num))

        self.world.tick()
        spectator = self.world.get_spectator()
        spectator.set_transform(self.ego.get_transform())

    def handle_ego(self):
        control = self.ego_behavior.run_step(self.actor,self.ego,self.ego_local_planner, self.world)
        self.ego.apply_control(control)

    def handle_actor(self):
        control = self.actor_behavior.run_step(self.actor,self.ego,self.actor_local_planner, self.world)
        self.actor.apply_control(control)

    def collect_data(self):
        # handle data collection for real time inference
        self.past_ego_buffer.append([self.ego.get_transform().location.x,self.ego.get_transform().location.y])
        self.past_actor_buffer.append([self.actor.get_transform().location.x,self.actor.get_transform().location.y])
        if len(self.past_ego_buffer) > 45:
            del(self.past_ego_buffer[0])
            del(self.past_actor_buffer[0])

        if COLLECTING and self.lidar_data is not None:
            self.collector.store_data(self.lidar_data,self.episode_num,self.ego.get_transform(),self.actor.get_transform())

        if PLANNER_TYPE == 2 and self.episode_num < 5 and self.constants.past_control_start(self.ego):
            self.all_trajectories[self.episode_num][0].append([self.ego.get_transform().location.x,self.ego.get_transform().location.y])
            self.all_trajectories[self.episode_num][1].append([self.actor.get_transform().location.x,self.actor.get_transform().location.y])

    def lidar_callback(self,data):
        self.lidar_data = data

    def collision_callback(self,event):
        self.collided = True

    def camera_callback(self,image):
        image.save_to_disk('Visualize/camera_folder/%.6d.jpg' % image.frame)

    def generate_sample(self):
        if len(self.past_ego_buffer) != 45:
            return

        ego_past = self.past_ego_buffer[2::3]
        actor_past = self.past_actor_buffer[2::3]
        future_batch = np.zeros(shape = (10,2,30,2))
        past_batch = np.stack([np.array([ego_past,actor_past]) for _ in range(10)],0)

        sensor_data = [[loc.x,loc.y,loc.z] for loc in self.lidar_data]
        bev = transform_lidar_data(np.array(sensor_data),60,360)
        bev = np.expand_dims(bev,axis = 2)[None]
        bev = np.tile(bev, (10, 1, 1, 1))

        yaws = np.array([self.ego.get_transform().rotation.yaw,self.actor.get_transform().rotation.yaw])
        yaws = np.stack([yaws for _ in range(10)],0)

        agent_presence = np.ones(shape=(10,2), dtype=np.float32)

        light_strings = np.tile(np.asarray("GREEN"), (10,))

        minibatch = self.inference.training_input.to_feed_dict(
            S_past_world_frame=past_batch.astype(np.float64),
            yaws=yaws.astype(np.float64),
            overhead_features=bev.astype(np.float64),
            agent_presence=agent_presence.astype(np.float64),
            S_future_world_frame=future_batch.astype(np.float64),
            light_strings=light_strings,
            metadata_list=interface.MetadataList(),
            is_training=np.array(False))
        sessrun = functools.partial(self.sess.run, feed_dict=minibatch)
        sampled_output_np = self.inference.sampled_output.to_numpy(sessrun)

        rollout = sampled_output_np.rollout.S_world_frame

        rollout = rollout[0] # choose first batch index
        for i,sample in enumerate(rollout): # 12 samples overall
            ego_traj = sample[0]
            for i in range(29):
                start = carla.Location(ego_traj[i][0],ego_traj[i][1],2)
                end = carla.Location(ego_traj[i+1][0],ego_traj[i+1][1],2)
                self.world.debug.draw_line(start, end, 0.1, carla.Color(255,0,0), life_time=1/30) # should only show for one frame

            actor_traj = sample[1]
            for i in range(29):
                start = carla.Location(actor_traj[i][0],actor_traj[i][1],2)
                end = carla.Location(actor_traj[i+1][0],actor_traj[i+1][1],2)
                self.world.debug.draw_line(start, end, 0.1, carla.Color(255,0,0), life_time=1/30) # should only show for one frame

    def evaluate_mode_prob(self):

        if len(self.past_ego_buffer) != 45:
            return

        yaws = np.array([self.ego.get_transform().rotation.yaw,self.actor.get_transform().rotation.yaw])
        yaws = np.stack([yaws for _ in range(10)],0)
        sensor_data = [[loc.x,loc.y,loc.z] for loc in self.lidar_data]
        bev = transform_lidar_data(np.array(sensor_data),60,360)
        bev = np.expand_dims(bev,axis = 2)[None]
        bev = np.tile(bev, (10, 1, 1, 1))
        yaw = np.deg2rad(self.ego.get_transform().rotation.yaw + 90)
        origin = np.array([self.ego.get_transform().location.x,self.ego.get_transform().location.y])
        R = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        world = np.dot(R,np.array(sensor_data)[:,:2].T).T + origin
        lidar_X = world[:,0].tolist()
        lidar_Y = world[:,1].tolist()
        agent_presence = np.ones(shape=(10,2), dtype=np.float32)
        light_strings = np.tile(np.asarray("GREEN"), (10,))

        episode = self.episode_num
        ego_future = []
        actor_future = []
        ego_past = np.array(self.past_ego_buffer)
        actor_past = np.array(self.past_actor_buffer)
        recording_future = False

        while True:

            if episode != self.episode_num:

                ego_future = np.array(ego_future)[:240][7::8]
                actor_future = np.array(actor_future)[:240][7::8]
                future_batch = np.tile(np.stack((ego_future,actor_future))[None],(10,1,1,1)) #(10,2,30,2)

                ego_past = np.array(ego_past)[-45:][2::3]
                actor_past = np.array(actor_past)[-45:][2::3]
                past_batch = np.stack([np.array([ego_past,actor_past]) for _ in range(10)],0)

                minibatch = self.inference.training_input.to_feed_dict(
                    S_past_world_frame=past_batch.astype(np.float64),
                    yaws=yaws.astype(np.float64),
                    overhead_features=bev.astype(np.float64),
                    agent_presence=agent_presence.astype(np.float64),
                    S_future_world_frame=future_batch.astype(np.float64),
                    light_strings=light_strings,
                    metadata_list=interface.MetadataList(),
                    is_training=np.array(False))

                perturb_ep = self.sess.graph.get_tensor_by_name('random_normal_2/stddev:0')
                minibatch[perturb_ep] = 0. # disable perturbation when we attempt to calculate log prior of given trajectory

                expert_log_prob = self.sess.run((self.tensor_collections['infer_output'][0]),minibatch)
                print("Average log prior value for this mode is %f" % np.mean(np.array(expert_log_prob)))

                fig, ax = plt.subplots(1,1, figsize=(50,50))
                ax.cla()
                ax.set_aspect('equal')
                ax.set_xlim(150,220)
                ax.set_ylim(-200,-150)
                ax.invert_xaxis()

                ego_future_x = np.array(ego_future)[:,0].tolist()
                ego_future_y = np.array(ego_future)[:,1].tolist()
                ax.plot(ego_future_x,ego_future_y,
                linestyle = '-',
                marker='o',
                markersize=3,
                zorder=1)

                actor_future_x = np.array(actor_future)[:,0].tolist()
                actor_future_y = np.array(actor_future)[:,1].tolist()
                ax.plot(actor_future_x,actor_future_y,
                linestyle = '-',
                marker='o',
                markersize=3,
                zorder=1)

                ego_past_x = np.array(ego_past)[:,0].tolist()
                ego_past_y = np.array(ego_past)[:,1].tolist()
                ax.plot(ego_past_x,ego_past_y,
                linestyle = '-',
                marker='o',
                markersize=3,
                zorder=1)

                actor_past_x = np.array(actor_past)[:,0].tolist()
                actor_past_y = np.array(actor_past)[:,1].tolist()
                ax.plot(actor_past_x,actor_past_y,
                linestyle = '-',
                marker='o',
                markersize=3,
                zorder=1)

                for i in range(30):
                    ax.plot([actor_future_x[i], ego_future_x[i]],[actor_future_y[i], ego_future_y[i]],
                    linestyle = '-',
                    lw = 0.5,
                    marker='o',
                    markersize=1,
                    zorder=1)

                ax.plot(lidar_X,lidar_Y,linestyle = 'None',marker = 'o',color='r',markersize=0.6,alpha=0.4)

                ego_future = []
                actor_future = []
                ego_past = []
                actor_past = []
                recording_future = False

                plt.show()

                episode = self.episode_num

            self.check_episode_end()
            self.handle_ego()
            self.handle_actor()

            self.world.tick()

            if recording_future or self.constants.past_control_start(self.ego):
                ego_future.append([self.ego.get_transform().location.x,self.ego.get_transform().location.y])
                actor_future.append([self.actor.get_transform().location.x,self.actor.get_transform().location.y])
                recording_future = True
            else:
                ego_past.append([self.ego.get_transform().location.x,self.ego.get_transform().location.y])
                actor_past.append([self.actor.get_transform().location.x,self.actor.get_transform().location.y])

    def plan(self):

        if len(self.past_ego_buffer) != 45:
            return

        ego_past = self.past_ego_buffer[2::3]
        actor_past = self.past_actor_buffer[2::3]
        future_batch = np.zeros(shape = (10,2,30,2))
        past_batch = np.stack([np.array([ego_past,actor_past]) for _ in range(10)],0)

        sensor_data = [[loc.x,loc.y,loc.z] for loc in self.lidar_data]
        bev = transform_lidar_data(np.array(sensor_data),60,360)
        bev = np.expand_dims(bev,axis = 2)[None]
        bev = np.tile(bev, (10, 1, 1, 1))

        yaws = np.array([self.ego.get_transform().rotation.yaw,self.actor.get_transform().rotation.yaw])
        yaws = np.stack([yaws for _ in range(10)],0)

        agent_presence = np.ones(shape=(10,2), dtype=np.float32)

        light_strings = np.tile(np.asarray("GREEN"), (10,))

        minibatch = self.inference.training_input.to_feed_dict(
            S_past_world_frame=past_batch.astype(np.float64),
            yaws=yaws.astype(np.float64),
            overhead_features=bev.astype(np.float64),
            agent_presence=agent_presence.astype(np.float64),
            S_future_world_frame=future_batch.astype(np.float64),
            light_strings=light_strings,
            metadata_list=interface.MetadataList(),
            is_training=np.array(False))

        if len(self.future_plan) == 0 or not CONTROL:
            if PLANNER_TYPE == 2:
                decision = self.planner.plan()
                self.ego_behavior.behavior_type = decision
                self.handle_ego()
                return
            else:
                traj,score = self.planner.plan(minibatch)
                print("Planned trajectory:")
                print(traj)

            ego_traj = traj[0]
            for i in range(30):
                loc = carla.Location(ego_traj[i][0],ego_traj[i][1],2)
                self.world.debug.draw_point(loc, 0.1, carla.Color(44,99,163,50), life_time = 8 * (REPLAN / 30)) # should only show for one frame
            for i in range(29):
                start = carla.Location(ego_traj[i][0],ego_traj[i][1],2)
                end = carla.Location(ego_traj[i+1][0],ego_traj[i+1][1],2)
                self.world.debug.draw_line(start, end, 0.05, carla.Color(44,99,163,50), life_time=8 * (REPLAN / 30)) # should only show for one frame

            actor_traj = traj[1]
            for i in range(30):
                loc = carla.Location(actor_traj[i][0],actor_traj[i][1],2)
                if CONTROL:
                    self.world.debug.draw_point(loc, 0.1, carla.Color(255,60,20,50), life_time = 8 * (REPLAN / 30)) # should only show for one frame
            for i in range(29):
                start = carla.Location(actor_traj[i][0],actor_traj[i][1],2)
                end = carla.Location(actor_traj[i+1][0],actor_traj[i+1][1],2)
                self.world.debug.draw_line(start, end, 0.05, carla.Color(255,60,20,50), life_time=8 * (REPLAN / 30)) # should only show for one frame


            self.future_plan = traj.tolist()[0]
            self.frame_count = 0
            self.vec_ref = [self.future_plan[-1][0] - self.future_plan[0][0],self.future_plan[-1][1] - self.future_plan[0][1]]


        if CONTROL:
            if self.constants.reached_goal(self.ego) or self.collided:
                self.reset()
                return

            # controllers
            self.throttle = ThrottleController(self.ego)
            self.steer = SteerController(self.ego)

            # we then handle control using the plan we have
            current_pos = self.past_ego_buffer[-1]
            target_pos = self.future_plan[0]
            target_pos_next = self.future_plan[1]
            current_speed = (self.ego.get_velocity().x ** 2 + self.ego.get_velocity().y ** 2) ** 0.5
            target_speed = ((target_pos[0] - target_pos_next[0])**2 + (target_pos[1] - target_pos_next[1]) ** 2) ** 0.5 / (8/30) * 1.105 # should be 3.6, used to be 1.105
            self.frame_count += 1

            # next check if we have reached eight frames, since each point takes eight frames to complete:
            if self.frame_count == 8:
                self.frame_count = 0
                self.previous_plan_point = self.future_plan[0]
                del self.future_plan[0]
                if len(self.future_plan) == 30 - REPLAN + 1:
                    self.future_plan = []

            if hasattr(self,'previous_plan_point'):
                vec_traj = [target_pos[0]-self.previous_plan_point[0],target_pos[1]-self.previous_plan_point[1]]
                if (vec_traj[0]**2 + vec_traj[1] ** 2) ** 0.5 < 0.20:
                    control = carla.VehicleControl()
                    control.throttle = 0
                    control.brake = 1
                    control.steer = 0
                    self.ego.apply_control(control)
                    return

            print("target speed: %f" % target_speed)
            print("current speed: %f" % current_speed)
            throttle = self.throttle.run_step(target_speed,current_speed)
            steer = self.steer.run_step(target_pos)
            need_to_brake = (target_speed - current_speed) < 0.

            if need_to_brake:
                brake = -throttle
                throttle = 0.0
            else:
                throttle = throttle
                brake = 0.0

            if target_speed < 0.3:
                brake = 1.0
                throttle = 0

            steer = max(-0.9, min(0.9, steer))
            brake = max(min(brake, 1.0), 0.0)
            throttle = min(max(throttle, 0.0), 1.0)
            hand_brake = False
            reverse = False

            control = carla.VehicleControl()
            control.throttle = throttle
            control.brake = brake
            control.steer = steer
            control.hand_brake = hand_brake
            control.reverse = reverse

            self.ego.apply_control(control)
            return

def main():
    runner = ScenarioRunner()
    try:
        runner.start()
    finally:

        def exit_handler():
            if RECORDING:
                image_folder = 'Visualize/camera_folder'
                video_name = 'Visualize/video.avi'
                images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
                frame = cv2.imread(os.path.join(image_folder, images[0]))
                height, width, layers = frame.shape
                video = cv2.VideoWriter(video_name, 0, 30,(width,height))
                print("Processing video...")
                for image in images:
                    video.write(cv2.imread(os.path.join(image_folder, image)))
                cv2.destroyAllWindows()
                video.release()
            if COLLECTING:
                runner.collector.compile_data()
        atexit.register(exit_handler)

        for a in runner.actor_list:
            a.destroy()

if __name__ == '__main__':
    main()
