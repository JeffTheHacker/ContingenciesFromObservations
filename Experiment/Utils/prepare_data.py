import sys
import attrdict
import json
sys.path.append('../../precog')
from imports import *
import tensorflow as tf

class LidarParams(attrdict.AttrDict):
    def __init__(self, meters_max=50, pixels_per_meter=2, hist_max_per_pixel=25, val_obstacle=1.):
        super().__init__(meters_max=meters_max,
                         pixels_per_meter=pixels_per_meter,
                         hist_max_per_pixel=hist_max_per_pixel)

class NumpyEncoder(json.JSONEncoder):
    """
    The encoding object used to serialize np.ndarrays
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dict_to_json(dict_datum, out_fn, b=0):
    """Used to serialize model feeds into json that can be used to train the model.

    :param dict_datum:
    :param out_fn:
    :returns:
    :rtype:

    """
    #assert(not os.path.isfile(out_fn))
    # Assume that the input dict has keys with .name attributes (e.g. tf.Tensors)
    preproc_dict = {k.name.split(':')[0]: np.squeeze(v[b]) for k, v in dict_datum.items()}
    with open(out_fn, 'w') as f:
        json.dump(preproc_dict, f, cls=NumpyEncoder)
    return out_fn

# constants
EPISODE_NUM = 500
B = 1
A = 1
T_past = 15
T = 30
H = 60
W = 360
C = 1
D = 2

# configurations
frames_to_ignore = 120
perturb_past = True
perturb_yaw = True

# Create Phi object
tf.compat.v1.reset_default_graph()
S_past_world_frame = tf.zeros((B, A, T_past, D), dtype=tf.float64, name="S_past_world_frame")
S_future_world_frame = tf.zeros((B, A, T, D), dtype=tf.float64, name="S_future_world_frame")
yaws = tf.zeros((B, A), dtype=tf.float64, name="yaws")
overhead_features = tf.zeros((B, H, W, C), dtype=tf.float64, name="overhead_features")
agent_presence = tf.zeros((B, A), dtype=tf.float64, name="agent_presence")
light_strings = tf.zeros((B,), dtype=tf.string, name="light_strings")
lidar_params = LidarParams()
is_training = tf.constant(False,dtype = tf.bool)
A_future_world_frame = tf.zeros((B, 1, T, D), dtype=tf.float64, name="A_future_world_frame")
A_past_world_frame = tf.zeros((B, 1, T_past, D), dtype=tf.float64, name="A_past_world_frame")
A_yaws = tf.zeros((B, 1), dtype=tf.float64, name="A_yaws")
phi = interface.ESPPhi(
    S_past_world_frame=S_past_world_frame,
    yaws=yaws,
    overhead_features=overhead_features,
    agent_presence=agent_presence,
    light_strings=light_strings,
    feature_pixels_per_meter=lidar_params.pixels_per_meter,
    is_training = is_training,
    yaws_in_degrees=True)

# process each episode
for i in range(EPISODE_NUM):#EPISODE_NUM):
    file_string = os.path.dirname(__file__) + "/../Data/Episode_%d" % (i)
    with open(file_string,'rb') as file:
        episode_data = pickle.load(file)

        frame_num = 0

        for frame in episode_data.frames:

            print("Processing episode %d and frame %d" % (i,frame_num))

            ego_yaw = frame[0]
            actor_yaw = frame[1]
            ego_past = frame[2]
            actor_past = frame[3]
            sensor_data = frame[4]
            ego_future = frame[5]
            actor_future = frame[6]

            # skipping condition for overtake
            # if ego_past[-1][1] < -53:
            #     frame_num += 1
            #     continue
            # skipping condition for left turn
            # if ego_past[-1][0] > 220:
            #     frame_num += 1
            #     continue
            # skipping condition for right turn
            # if ego_past[-1][1] < -191:
            #     frame_num += 1
            #     continue


            # take every several rows of ego_past and ego_future, and same for actor
            ego_past = ego_past[2::3]
            ego_future = ego_future[7::8]
            actor_past = actor_past[2::3]
            actor_future = actor_future[7::8]

            # add perturbations
            if perturb_past:
                ego_past += np.random.normal(0,0.02,size = (15,3))
                actor_past += np.random.normal(0,0.02,size = (15,3))

            # Extract pasts by taking only x and y values, and adding a dimension
            ego_past = ego_past[:,:2][None]
            actor_past = actor_past[:,:2][None]

            # Tile pasts.
            ego_pasts_batch = ego_past[None]

            # fill in yaws
            if perturb_yaw:
                ego_yaw = np.random.normal(ego_yaw,1.5)
                if ego_yaw < -180:
                    ego_yaw = 360 + ego_yaw
                if ego_yaw > 180:
                    ego_yaw = ego_yaw - 360
            ego_yaws = np.tile(np.array([ego_yaw])[None],(B,1))

            # Indicate all present
            agent_presence = np.ones(shape=tensoru.shape(phi.agent_presence), dtype=np.float32)

            feed_dict = tfutil.FeedDict()
            feed_dict[phi.S_past_world_frame] = ego_pasts_batch
            feed_dict[phi.yaws] = ego_yaws
            feed_dict[phi.agent_presence] = agent_presence

            # process lidar data
            bev = transform_lidar_data(sensor_data,H,W)
            bev = np.expand_dims(bev,axis = 2)[None]
            bev = np.tile(bev, (B, 1, 1, 1))

            # Add the bevs to the feed dict.
            feed_dict[phi.overhead_features] = bev

            # mask our traffic light data
            light_string_batch = np.tile(np.asarray("GREEN"), (B,))
            feed_dict[phi.light_strings] = light_string_batch

            # add future prediction data
            ego_future = ego_future[:,:2][None]
            actor_future = actor_future[:,:2][None]
            feed_dict[S_future_world_frame] = ego_future[None]
            feed_dict[A_future_world_frame] = actor_future[None]
            feed_dict[A_past_world_frame] = actor_past[None]
            feed_dict[A_yaws] = np.tile(np.array([actor_yaw])[None],(B,1))

            # save the feed_dict
            fn = os.path.dirname(__file__) + "/../Data/JSON_output/feed_Episode_%d_frame_%d.json" % (i,frame_num)
            dict_to_json(feed_dict, fn)

            frame_num += 1
