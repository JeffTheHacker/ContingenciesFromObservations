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
parser.add_argument('-cp', '--checkpoint_path', action='store', dest='checkpoint_path', type=str, help='Absolute path to model checkpoint, example: ...SocialConvRNN_/esp-model-100000', default='../Model/esp_train_results/2021-01/01-24-20-31-06_Left_Turn_Dataset_precog.bijection.social_convrnn.SocialConvRNN_/esp-model-668000')
parser.add_argument('-mp', '--model_path', action='store', dest='model_path', type=str, help='Absolute path to model folder, example: ...SocialConvRNN_', default='../Model/esp_train_results/2021-01/01-24-20-31-06_Left_Turn_Dataset_precog.bijection.social_convrnn.SocialConvRNN_')
parser.add_argument('--enable-inference', dest='inference',action='store_true', help='Run planning and plot trajectories in Carla.')
parser.add_argument('--enable-control', dest='control',action='store_true', help='Control the ego vehicle with planned trajectories.')
parser.add_argument('-r', '--replan', action='store', type=int, dest='replan', help='Integer > 0. Sets replanning rate. Smaller -> more frequent replanning.')
parser.add_argument('-t', '--planner_type', action='store', type=int, dest='planner_type', help='Sets the type of planner. 0: contingent, 1: overconfident, 2: underconfident.')
parser.add_argument('-s', '--scenario', action='store', type=int, dest='scenario', help='Sets the scenario. 0: left turn, 1: overtake, 2: right turn')
parser.add_argument('-l', '--location', action='store', type=int, dest='location', help='Sets location of the scenario, an integer from 0 to 3.')
parser.add_argument('--video_name', type=str, dest='video_name', default='video', help='Video name if recording')
parser.add_argument('--max_episodes', type=int, dest='max_episodes', default=None, help='Max number of episodes to run before terminating')

# MFP args
parser.add_argument('--mfp_control', dest='mfp_control', action='store_true', help='Whether or not to use MFP for control')
parser.add_argument('--mfp_planning_choice', type=str, dest='mfp_planning_choice', default='highest_score_weighted', choices=['highest_score_threshold', 'highest_score_weighted', 'highest_prob'], help='Planning method for MFP')
parser.add_argument('--mfp_checkpoint', type=str, dest='mfp_checkpoint', default=None, help='MFP checkpoint to load')
parser.add_argument('--mfp_checkpoint_itr', type=str, dest='mfp_checkpoint_itr', default='latest', help='Specific save point to load')

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

# Import packages from MFP repo
import os
assert 'PRECOGROOT' in os.environ, "Make sure to run `source precog_env.sh`"
ROOT_DIR = os.environ['PRECOGROOT']
assert 'MFPROOT' in os.environ, "Set MFPROOT to point to the MFP directory!"
MFP_ROOT = os.environ['MFPROOT']
MFP_CKPT = os.path.join(MFP_ROOT, 'multiple_futures_prediction/checkpts', args.mfp_checkpoint)
MFP_CKPT_ITR = args.mfp_checkpoint_itr
MFP_CONTROL = args.mfp_control
MFP_PLANNING_CHOICE = args.mfp_planning_choice

sys.path.append(MFP_ROOT)
from multiple_futures_prediction.model_carla import mfpNet
from multiple_futures_prediction.train_carla import Params
from multiple_futures_prediction.dataset_carla import CarlaDataset
from multiple_futures_prediction.demo_carla import load_mfp_model 
from multiple_futures_prediction.my_utils import rotate_np 

def plt_setup():
    if SCENARIO == 0: # left turn
        PADDING = 50
        plt.xlim(160,230+PADDING)
        plt.ylim(-180,-150+PADDING)
    elif SCENARIO == 1: # overtake
        PADDING = 10
        plt.xlim(-90-PADDING,-85+PADDING)
        plt.ylim(-60-PADDING,5+6*PADDING) # needs extra padding in direction both cars are moving
    elif SCENARIO == 2: # right turn
        PADDING = 25
        plt.xlim(160-3*PADDING,260) # needs extra padding in direction ego's turning / actor's moving
        plt.ylim(-210,-170+PADDING)

def ax_setup(ax):
    if SCENARIO == 0: # left turn
        PADDING = 50
        ax.set_xlim(160,230+PADDING)
        ax.set_ylim(-180,-150+PADDING)
    elif SCENARIO == 1: # overtake
        PADDING = 10
        ax.set_xlim(-90-PADDING,-85+PADDING)
        ax.set_ylim(-60-PADDING,5+6*PADDING) # needs extra padding in direction both cars are moving
    elif SCENARIO == 2: # right turn
        PADDING = 25
        ax.set_xlim(160-3*PADDING,260) # needs extra padding in direction ego's turning / actor's moving
        ax.set_ylim(-210,-170+PADDING)


class ScenarioRunner(object):

    def obs_to_mfp_input(self, ego_past, actor_past, ego_yaw, actor_yaw, im_crop=None):
        """Turn the gym observation into the input format
        used with the (pytorch) MFP model"""
        ego_past = np.array(ego_past)
        actor_past = np.array(actor_past)
        ego_fut = np.zeros((30,2))
        actor_fut = np.zeros((30,2))
        # NOTE: this will shift the future to start at 0, but in our case it shouldn't matter
        #       (since we're passing 0s/dummy values in)
        hist, fut = self.mfp_carla_dataset.process_xy_data(
          ego_past, ego_fut, ego_yaw,
          actor_past, actor_fut, actor_yaw)
        # neighbors[ind] maps from
        #     0-index'd vehicle_id
        # to
        #     list of (0-index'd neighbor's vehicle_id, neighbor's vehicle_id, neighbor's grid pos)
        neighbors = {
          0: [(1, 1, 1)],
          1: [(0, 0, 0)],
        }

        # yaw = np.deg2rad(self.ego.get_transform().rotation.yaw + 90)
        yaws = np.array([np.deg2rad(ego_yaw), np.deg2rad(actor_yaw)])

        ### (2) Preprocess into "batch mode"
        hist, nbrs, mask, fut, mask, context, yaws, nbrs_info = \
          self.mfp_carla_dataset.collate_fn([(hist, fut, neighbors, im_crop, yaws)])
        
        ### (3) Do required preprocessing (offsetting)
        if self.mfp_params.remove_y_mean:
          # preprocess the (real) future traj before passing to model
          fut = fut-y_mean.unsqueeze(1)

        if self.mfp_params.use_cuda:
          hist = hist.cuda()
          nbrs = nbrs.cuda()
          mask = mask.cuda()
          fut = fut.cuda()
          if context is not None:
            context = context.cuda()
          if yaws is not None:
            yaws = yaws.cuda()

        return hist, nbrs, mask, fut, mask, context, yaws, nbrs_info

    def mfp_output_to_trajs(self, ref_pos, yaws, fut_preds, modes_pred):
        """Post-process MFP outputs into easy to use trajectories"""
        assert len(modes_pred) == 2, modes_pred
        assert all([np.isclose(1,i.cpu().detach().numpy().sum()) for i in modes_pred]), \
          [i.cpu().detach().numpy().sum() for i in modes_pred]
        for i,m in enumerate(modes_pred):
            m = m.cpu().detach().numpy()

        ego_fut_probs = modes_pred[0].cpu().detach().numpy() # (2,)
        assert np.isclose(1,ego_fut_probs.sum()), ego_fut_probs.sum()
        actor_fut_probs = modes_pred[1].cpu().detach().numpy() # (2,)
        assert np.isclose(1,actor_fut_probs.sum()), actor_fut_probs.sum()

        # for each mode (eg, K=25), a tuple of (horizon, n_agents, mean/std/...)
        assert all([i.shape == (30,2,5) for i in fut_preds]), fut_preds[0].shape

        ego_trajs = []
        actor_trajs = []
        for k in range(len(fut_preds)):
          fut_preds_mode_k = fut_preds[k].cpu().detach().numpy()[:,:,:2]
          # preprocess - add back mean
          if self.mfp_params.remove_y_mean:
            fut_preds_mode_k += y_mean.unsqueeze(1).cpu().detach().numpy() # 30,2,2
          # preprocess - add back reference position
          # NOTE offset by final hist pos
          fut_preds_mode_k += ref_pos.view(1,-1,2).cpu().detach().numpy() 

          if self.mfp_params.rotate_pov:
            # Need to rotate the generated futures 
            # AND the "ground truth" futures since the
            # histories which were fed as inputs to the model were
            # rotated to POV view
            ref_pos_np = ref_pos.cpu().detach().numpy()
            fut_preds_mode_k[:,0,:] = rotate_np(
              ref_pos_np[0,:],
              fut_preds_mode_k[:,0,:],
              -yaws[0],
              degrees=False)
            fut_preds_mode_k[:,1,:] = rotate_np(
              ref_pos_np[1,:],
              fut_preds_mode_k[:,1,:],
              -yaws[1],
              degrees=False)

          ego_trajs.append( (ego_fut_probs[k], fut_preds_mode_k[:,0,:]) )
          actor_trajs.append( (actor_fut_probs[k], fut_preds_mode_k[:,1,:]) )
          
        return (ego_trajs, actor_trajs)

    def draw_traj(self, traj, color, point_size=0.1, line_size=0.05,
                  life_time=8*(REPLAN/30),
                  x_offset=0, y_offset=0, z_offset=0,
                  text=None, text_ts=0, text_color=None):
        life_time = 1/30
        for i in range(traj.shape[0]): # dots
            loc = carla.Location(traj[i][0]+x_offset, traj[i][1]+y_offset, 2+z_offset)
            self.world.debug.draw_point(loc, point_size, 
              color, life_time=life_time) # should only show for one frame
        for i in range(traj.shape[0]-1): # lines
            start = carla.Location(traj[i][0]+x_offset, traj[i][1]+y_offset, 2+z_offset)
            end = carla.Location(traj[i+1][0]+x_offset, traj[i+1][1]+y_offset, 2+z_offset)
            self.world.debug.draw_line(start, end, line_size,
              color, life_time=life_time) # should only show for one frame
        if text:
            if not text_color:
                text_color = color
            self.world.debug.draw_string(
              carla.Location(traj[text_ts][0]+x_offset,traj[text_ts][1]+y_offset,2+z_offset),
              text, color=text_color,
              # NOTE: for some reason, -1 is the only arg that seems to avoid
              # duplicate text annotations in the simulator
              life_time=-1)

    def start(self):

        # Try to load MFP model before doing anything CARLA-related
        print("Loading MFP model...")
        self.mfp_net, self.mfp_params, ckpt_file = load_mfp_model(MFP_CKPT,checkpoint=MFP_CKPT_ITR)
        print("...done")

        print("Loading MFP CarlaDataset object (for preprocessing model inputs)...")
        d_s = self.mfp_params.subsampling
        t_h = self.mfp_params.hist_len_orig_hz
        t_f = self.mfp_params.fut_len_orig_hz
        self.mfp_carla_dataset = CarlaDataset(
          os.path.join(MFP_ROOT, 'multiple_futures_prediction/carla_data_cfd/Left_Turn_Dataset/train'),
          t_h, t_f, d_s, self.mfp_params.encoder_size, self.mfp_params.use_gru, self.mfp_params.self_norm,
          self.mfp_params.data_aug, self.mfp_params.use_context, self.mfp_params.nbr_search_depth,
          shuffle=False)
        self._plan_counter = 0
        print("...done")

        self.VIS_DIR = "{}-{}".format(os.path.basename(MFP_CKPT), os.path.basename(ckpt_file))
        if RECORDING:
            image_folder = os.path.join(ROOT_DIR,'Experiment/Visualize/camera_folder_{}'.format(self.VIS_DIR))
            if os.path.isdir(image_folder):
                print("Warning: dir %s already exists!" % image_folder)
            else:
                # create
                os.mkdir(image_folder)
            images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
            # Wipe previous images
            for im in images:
                os.remove(os.path.join(image_folder,im))

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

            if args.max_episodes and self.episode_num >= args.max_episodes:
                print("Reached max episodes, exiting..")
                raise KeyboardInterrupt

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

        # NOTE: Need a copy of the cost model to turn MFP forecasting into planning
        self.mfp_cost_model = CostedGoalLikelihood(
          None, self.constants, batch_dim=1, traj_dim=self.mfp_params.modes**2)
        self.mfp_cost_model_ph = tf.placeholder(
          'float64',shape=(1,self.mfp_params.modes**2,2,30,2)) # xy, T, agents
        self.mfp_cost_model_op = self.mfp_cost_model.log_prob(self.mfp_cost_model_ph)

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
        image.save_to_disk(
          os.path.join(ROOT_DIR,
                       'Experiment/Visualize/camera_folder_{}'.format(self.VIS_DIR),
                       '%.6d.jpg' % image.frame))

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
        self._plan_counter += 1

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

        if not MFP_CONTROL:
            minibatch = self.inference.training_input.to_feed_dict(
                S_past_world_frame=past_batch.astype(np.float64),
                yaws=yaws.astype(np.float64),
                overhead_features=bev.astype(np.float64),
                agent_presence=agent_presence.astype(np.float64),
                S_future_world_frame=future_batch.astype(np.float64),
                light_strings=light_strings,
                metadata_list=interface.MetadataList(),
                is_training=np.array(False))


        ### Take traj data and feed it into the MFP model

        if self.mfp_params.use_context:
            mfp_bev = transform_lidar_data(np.array(sensor_data),60,360)
            context = mfp_bev.astype(np.float64)[np.newaxis,:,:]
            assert context.shape == (1,60,360), context.shape
        else:
            context = None
        hist, nbrs, mask, fut, mask, context, yaws, nbrs_info = \
          self.obs_to_mfp_input(
            ego_past, actor_past,
            self.ego.get_transform().rotation.yaw,
            self.actor.get_transform().rotation.yaw,
            context)
        bStepByStep = True
        if self.mfp_params.no_atten_model:
          fut_preds, modes_pred = self.mfp_net.forward_mfp(
            hist, nbrs, mask, context,
            nbrs_info, fut, bStepByStep,
            yaws=yaws, rotate_hist=self.mfp_params.rotate_pov)
        else:
          fut_preds, modes_pred = self.mfp_net.forward_mfp(
            hist, nbrs, mask, context,
            nbrs_info, fut, bStepByStep)

        # re-arrange into easy-to-use format
        ref_pos = hist[-1,:,:] # final (i=0) x,y (i=2) for all vehicles (i=1)
        ego_futures, actor_futures = self.mfp_output_to_trajs(
          ref_pos,
          yaws.cpu().detach().numpy(),
          fut_preds, modes_pred)

        if len(self.future_plan) == 0 or not CONTROL:
            if not MFP_CONTROL:
                if PLANNER_TYPE == 2:
                    decision = self.planner.plan()
                    self.ego_behavior.behavior_type = decision
                    self.handle_ego()
                    return
                else:
                    traj,score = self.planner.plan(minibatch)
                    print("Planned trajectory:")
                    print(traj)


            # Keep track of the highest scoring future that MFP generated
            # We want to use that one for planning
            max_joint_traj_score = None
            best_plan = None

            assert len(ego_futures) == len(actor_futures)

            ### Find the best trajectory to use for planning with a PID waypoint controller

            # Exhaustive search, K^2
            # NOTE: this can be batched, no need for a nested loop
            joint_traj_all = np.zeros((self.mfp_params.modes**2,2,30,2),dtype='float64')

            # Do we want to ignore trajs under a certain threshold?
            #planning_p_threshold = 0.01
            planning_p_threshold = 0.001
            traj_idxs_to_ignore = []
            traj_weights = []

            traj_idx = 0
            for (ego_prob, ego_traj) in ego_futures:
              ego_traj = ego_traj.astype('float64')  
              for (actor_prob, actor_traj) in actor_futures:
                actor_traj = actor_traj.astype('float64')
                # Fill the batched array to send to tf
                joint_traj_all[traj_idx,0,:,:] = ego_traj
                joint_traj_all[traj_idx,1,:,:] = actor_traj
                # Don't consider low prob trajectories
                if MFP_PLANNING_CHOICE == 'highest_score_threshold':
                  if (ego_prob < planning_p_threshold) or (actor_prob < planning_p_threshold):
                    traj_idxs_to_ignore.append(traj_idx)
                if MFP_PLANNING_CHOICE == 'max_min_score_threshold':
                  # NOTE: only threshold ego but not actor
                  if ego_prob < planning_p_threshold:
                    traj_idxs_to_ignore.append(traj_idx)
                # Weight trajectories by "joint" probability
                if MFP_PLANNING_CHOICE == 'highest_score_weighted':
                  traj_weights.append(ego_prob * actor_prob)
                # Move counter
                traj_idx += 1
                
            # Visualize all the individual futures before scoring
            plt.clf()
            for i, (ego_prob, ego_traj) in enumerate(ego_futures):
              ego_traj = ego_traj.astype('float64')  
              plt.scatter(ego_traj[:,0],ego_traj[:,1],alpha=0.5,label='ego{}, p={:.4f}'.format(i,ego_prob))
            for i, (actor_prob, actor_traj) in enumerate(actor_futures):
              actor_traj = actor_traj.astype('float64')
              plt.scatter(actor_traj[:,0],actor_traj[:,1],alpha=0.5,label='actor{}, p={:.4f}'.format(i,actor_prob))
            plt_setup()
            plt.gca().invert_xaxis() 
            plt.legend()
            plt.title('All futures (ego and actor)')
            plt.savefig('{}_all_futures.png'.format(self._plan_counter))

            # Feed the trajectories through the TF cost model
            joint_traj_all = joint_traj_all[np.newaxis,:,:,:,:]
            assert joint_traj_all.shape == (1,self.mfp_params.modes**2,2,30,2)
            joint_traj_all_scores = self.sess.run(
              self.mfp_cost_model_op,
              feed_dict={self.mfp_cost_model_ph: joint_traj_all})
            assert joint_traj_all_scores.shape == (1,self.mfp_params.modes**2)
            joint_traj_all_scores = joint_traj_all_scores[0]

            if MFP_PLANNING_CHOICE == 'highest_score_threshold':
              # Before choosing the best joint traj, wipe the scores of the
              # joint trajs where either ego or actor were low probability
              print("Preprocessed scores max/min=",joint_traj_all_scores.max(),joint_traj_all_scores.min())
              print("Setting %d/%d total joint traj with either ego/actor low prob" %
                    (len(traj_idxs_to_ignore), self.mfp_params.modes**2))
              joint_traj_all_scores[traj_idxs_to_ignore] = np.NINF # negative inf
              print("Clipped scores max/min=",joint_traj_all_scores.max(),joint_traj_all_scores.min())
              # Extract the ego traj from the highest scoring joint traj
              best_plan_idx = np.argmax(joint_traj_all_scores)
              best_joint_ego = joint_traj_all[0,best_plan_idx,0,:,:] # (1,trajs,agents,T,xy)
              best_joint_actor = joint_traj_all[0,best_plan_idx,1,:,:] # (1,trajs,agents,T,xy)
              best_plan = best_joint_ego # NOTE: ego is being used to plan
            elif MFP_PLANNING_CHOICE == 'highest_score_weighted':
              # Before choosing the best joint traj, weight the scores by p_ego * p_actor
              print("Preprocessed scores max/min=",joint_traj_all_scores.max(),joint_traj_all_scores.min())
              joint_traj_all_scores = joint_traj_all_scores * traj_weights
              # Extract the ego traj from the highest scoring joint traj
              print("Weighted scores max/min=",joint_traj_all_scores.max(),joint_traj_all_scores.min())
              best_plan_idx = np.argmax(joint_traj_all_scores)
              best_joint_ego = joint_traj_all[0,best_plan_idx,0,:,:] # (1,trajs,agents,T,xy)
              best_joint_actor = joint_traj_all[0,best_plan_idx,1,:,:] # (1,trajs,agents,T,xy)
              best_plan = best_joint_ego # NOTE: ego is being used to plan


            ### Visualize all the joint trajectories after scoring
            # This plots one ego per-FIGURE, with all actors collated

            traj_idx = 0
            for e_i, (ego_prob, ego_traj) in enumerate(ego_futures):
              ego_traj = ego_traj.astype('float64')  

              plt.clf()
              fig, ax = plt.subplots(1,1)
              axn = ax
              ax_setup(axn)
              axn.invert_xaxis()
              axn.set_title('ego {}, all possible joints'.format(e_i))
              axn.tick_params(axis='both', which='major', labelsize=8)
              axn.set_aspect('equal')

              axn.scatter(ego_traj[:,0],ego_traj[:,1],alpha=0.5,label='ego{}, p={:.2f}'.format(e_i,ego_prob))
              for a_i, (actor_prob, actor_traj) in enumerate(actor_futures):
                actor_traj = actor_traj.astype('float64')
                # Plot the joint with its score and joint probability
                axn.scatter(actor_traj[:,0],actor_traj[:,1],alpha=0.5,
                  label='actor{}, p={:.4f}, p_j={:.4f}, s={:.4f}'.format(
                    a_i,actor_prob,ego_prob*actor_prob,joint_traj_all_scores[traj_idx]))
                traj_idx += 1

              axn.legend()
              fig.tight_layout()
              plt.savefig('{}_ego{}_joints.png'.format(self._plan_counter,e_i))

            if MFP_CONTROL and MFP_PLANNING_CHOICE != 'highest_prob':
                # self.future_plan is used by the waypoint follower
                self.future_plan = best_joint_ego.tolist()
                # Can use this to inspect visualizations frame-by-frame
                #input("pause")

                # Visualize the "best" joint trajectory
                color_purple = carla.Color(169,3,252,10) # purple for ego
                color_yellow = carla.Color(252,223,3,10) # yellow for actor
                self.draw_traj(best_joint_ego, color_purple, point_size=0.05, line_size=0.025,
                  text="best_joint_ego, score={:.2f}".format(joint_traj_all_scores[best_plan_idx]),
                  text_ts=0)
                self.draw_traj(best_joint_actor, color_yellow, point_size=0.05, line_size=0.025,
                  text="best_joint_actor, score={:.2f}".format(joint_traj_all_scores[best_plan_idx]),
                  text_ts=0)

            ### Visualize all the trajectories with a p > thresh
            viz_p_threshold = planning_p_threshold
            viz_counter_ego = 0
            viz_counter_actor = 0
            for (ego_prob, ego_traj), \
                (actor_prob, actor_traj) in \
                zip(ego_futures, actor_futures):

              assert ego_traj.shape == (30,2)
              assert actor_traj.shape == (30,2)
              # note: needs to be f64 (not f32) or CARLA throws error
              ego_traj = ego_traj.astype('float64')  
              actor_traj = actor_traj.astype('float64')

              if ego_prob > viz_p_threshold:
                # Highest prob traj gets a different color
                if ego_prob == max([p for p,t in ego_futures]):
                    ego_color = carla.Color(44,99,163,10) # blue
                    ego_text="ego_traj (max p), p={:.2f}".format(ego_prob)
                    print("ego_p = %.2f, max = dark blue" % ego_prob)
                    # NOTE: we can use the highest prob traj for control
                    if MFP_CONTROL and MFP_PLANNING_CHOICE == 'highest_prob':
                        self.future_plan = ego_traj.tolist()
                else:
                    print("ego_p = %.2f, non-max = light blue" % ego_prob)
                    ego_color = carla.Color(36,214,214,10) # light blue
                    ego_text="ego_traj, p={:.2f}".format(ego_prob)
                print("mfp ego traj",type(ego_traj),ego_traj.shape,ego_traj.dtype)
                self.draw_traj(ego_traj, ego_color,
                  z_offset=.2*(viz_counter_ego+1),
                  text=ego_text,
                  text_ts=min(ego_traj.shape[0]-1, 1 + viz_counter_ego*1))
                viz_counter_ego += 1


              if actor_prob > viz_p_threshold:
                if actor_prob == max([p for p,t in actor_futures]):
                    print("actor_p = %.2f, max = red orange" % actor_prob)
                    actor_color = carla.Color(255,60,20,10) # red/orange
                else:
                    print("actor_p = %.2f, non-max = dark red" % actor_prob)
                    actor_color = carla.Color(207,0,0,10) # dark red
                self.draw_traj(actor_traj, actor_color,
                  z_offset=.2*(viz_counter_actor+1),
                  text="actor_traj, p={:.2f}".format(actor_prob),
                  text_ts=min(actor_traj.shape[0]-1, 1*(viz_counter_actor+1)))
                viz_counter_actor += 1

            #self.future_plan = traj.tolist()[0]
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
                image_folder = os.path.join(ROOT_DIR,'Experiment/Visualize/camera_folder_{}'.format(runner.VIS_DIR))
                video_name = os.path.join(ROOT_DIR,'Experiment/Visualize/video_{}.avi'.format(runner.VIS_DIR))
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
