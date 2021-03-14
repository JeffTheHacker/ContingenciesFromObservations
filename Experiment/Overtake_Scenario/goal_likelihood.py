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
        last_pos = s_1[:,:,-1,1] # (10,12)
        self.tp_prob = last_pos

        # crash penalty
        dis = (trajectories[:,:,0] - trajectories[:,:,1]) ** 2 # 10,12,30,2
        dis = dis ** 0.5 # 10,12,30,2
        dis_x = dis[:,:,:,0]
        dis_y = dis[:,:,:,1]
        distance_x = tf.constant([1],dtype = np.float64) # 8
        distance_x = tf.tile(distance_x[None,None],(10,12,30))
        criterion_1 = tf.sign(tf.nn.relu(distance_x - dis_x))
        distance_y = tf.constant([8.0],dtype = np.float64) # 8
        distance_y = tf.tile(distance_y[None,None],(10,12,30))
        criterion_2 = tf.sign(tf.nn.relu(distance_y - dis_y))
        pos_y_ego = trajectories[:,:,0,:,1] #10,12,30
        pos_y_actor = trajectories[:,:,1,:,1] #10,12,30
        criterion_3 = tf.sign(tf.nn.relu(pos_y_ego - pos_y_actor))
        result = - criterion_1 * criterion_2 * criterion_3 * 1000
        self.cp_prob = tf.reduce_min(result,axis = -1) #sum

        return self.cp_prob + self.tp_prob

    def describe(self):
        return "Cost Map Goal Likelihood"
