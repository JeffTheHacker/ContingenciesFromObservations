import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np 

class CostedGoalLikelihood:
    # time penalty, collision penalty, intersection penalty

    def __init__(self, model, constants):
        self.constants = constants

    def log_prob(self, trajectories):

        # time penalty
        s_1 = trajectories[:, :, 0] #(10,12,30,2)
        last_pos = s_1[:,:,-1] # (10,12,2)
        first_pos = s_1[:,:,0]
        self.tp_prob = tf.reduce_sum((first_pos - last_pos) ** 2, axis = -1) #10,12
        self.tp_prob = self.tp_prob ** 0.5
        self.tp_prob *= 6.5

        # crash penalty
        dis = (trajectories[:,:,0] - trajectories[:,:,1]) ** 2 # 10,12,30,2
        dis = tf.reduce_sum(dis,axis = -1)
        dis = dis ** 0.5 # 10,12,30
        distance = tf.constant([8],dtype = np.float64) # 8
        distance = tf.tile(distance[None,None],(10,12,30))
        criterion_1 = tf.nn.relu(distance - dis)
        x_cor = trajectories[:,:,0,:,0] # 10,12,30
        threshold_x = tf.constant([self.constants.ego_spawn_loc.location.x - 22],dtype = np.float64) # 207
        threshold_x = tf.tile(threshold_x[None,None],(10,12,30))
        criterion_2 = tf.sign(tf.nn.relu((threshold_x - x_cor)))
        y_cor = trajectories[:,:,0,:,1] # 10,12,30
        threshold_y = tf.constant([self.constants.ego_spawn_loc.location.y + 7],dtype = np.float64) # 207
        threshold_y = tf.tile(threshold_y[None,None],(10,12,30))
        criterion_3 = tf.sign(tf.nn.relu((threshold_y - y_cor)))
        result = - criterion_1 * criterion_2 * criterion_3 * 1000
        self.cp_prob = tf.reduce_sum(result,axis = -1) #sum

        return self.cp_prob + self.tp_prob

    def describe(self):
        return "Cost Map Goal Likelihood"
