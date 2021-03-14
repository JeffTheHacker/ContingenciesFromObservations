import tensorflow as tf
import tensorflow_probability as tfp
import precog.utils.np_util as npu
import precog.utils.class_util as classu
import precog.utils.tensor_util as tensoru
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
log = logging.getLogger(os.path.basename(__file__))


class PosteriorComponents:
    @classu.member_initialize
    def __init__(self, log_prior, log_goal_likelihood, log_posterior, dlog_prior_dz, dlog_goal_likelihood_dz, dlog_posterior_dz):
        """Stores posterior inference targets and some gradients of the posterior.

        :param log_prior:
        :param log_goal_likelihood:
        :param log_posterior:
        :param dlog_prior_dz:
        :param dlog_goal_likelihood_dz:
        :param dlog_posterior_dz:
        :returns:
        :rtype:

        """
        self.inference_targets = (self.log_prior, self.log_goal_likelihood, self.log_posterior)
        self.gradient_step_targets = self.inference_targets + (self.dlog_posterior_dz,)

    def validate_shape(self, objective_shape):
        # Ensure each term matches our shape expectations.
        assert(tensoru.shape(self.log_goal_likelihood) == objective_shape)
        assert(tensoru.shape(self.log_prior) == objective_shape)
        assert(tensoru.shape(self.log_posterior) == objective_shape)

class ContingentPlanner:
    # optimize over only robot z and resampling human z at every timestep
    @classu.member_initialize
    def __init__(self, model, goal_likelihood, sess):
        """ Stores the model, constructs the posterior, enables gradient-ascent on the posterior

        :param model:
        :param goal_likelihood:
        :param sess:
        :param check_gradients: whether to check gradients exist.

        """

        log.info("Instantiating {}.".format(self.__class__.__name__))
        # --- Define some useful functions.
        # (B, K, A, T, D) -> (B, K, T, D)
        partial_derivative = lambda t, dv: tf.gradients(t, dv)[0]

        # --- Extract and shape the trajectories and planning variables from the model.
        sampled_output = model.sampled_output
        sampled_rollout = sampled_output.rollout
        self.objective_shape = (10,12)

        # (10, 12, 2, 30, 2)
        self.Z = sampled_output.base_and_log_q.Z_sample
        # (10, 12, 2, 30, 2)
        self.S_world_frame = sampled_rollout.S_world_frame
        # (10,12)
        log_prior = tf.identity(sampled_output.base_and_log_q.log_q_samples, "log_prior")

        # --- Compute the log goal likelihood and form the planning criterion.
        # (10,12)
        goal_log_likelihood_values = tf.identity(goal_likelihood.log_prob(self.S_world_frame), "log_goal_likelihoods")
        # (10,12)
        #log_posteriors = tf.identity(log_prior + goal_log_likelihood_values, "log_posteriors")
        log_posteriors = tf.identity(goal_log_likelihood_values, "log_posteriors") #changed

        # --- Compute the gradients.
        log.info("Computing planning log-posterior gradients...")
        # (10, 12, 2, 30, 2)
        #dlogq_dz = tf.identity(partial_derivative(log_prior, self.Z), "dlogq_dz")
        dlogq_dz = tf.zeros_like(tf.identity(partial_derivative(log_prior, self.Z), "dlogq_dz")) #changed

        if goal_likelihood.describe() == "EmptyGoalLikelihood": dloggoal_dz = tf.zeros_like(dlogq_dz)
        else: dloggoal_dz = tf.identity(partial_derivative(goal_log_likelihood_values, self.Z),"dloggoal_dz")

        # (10, 12, 2, 30, 2)
        #dlog_posterior_dz = tf.identity(dlogq_dz + dloggoal_dz, "dlog_posterior_dz")
        dlog_posterior_dz = tf.identity(dloggoal_dz, "dlog_posterior_dz") #changed

        # -- Package the components of the posterior.
        self.posterior_components = PosteriorComponents(
            log_prior=log_prior,
            log_goal_likelihood=goal_log_likelihood_values,
            log_posterior=log_posteriors,
            dlog_prior_dz=dlogq_dz,
            dlog_goal_likelihood_dz=dloggoal_dz,
            dlog_posterior_dz=dlog_posterior_dz)
        log.info("Done computing gradients.")
        self.posterior_components.validate_shape(self.objective_shape)
        log.info("Done instantiating {}.".format(self.__class__.__name__))

        self.times = 0
        self.saved = None

    def plan(self,feed_dict, n = 100, epochs = 1):

        self.times += 1

        if self.Z in feed_dict:
            del feed_dict[self.Z]

        best_x_planned = None
        best_x_planned_posterior = -float('inf')

        for k in range(epochs):
            log.info("Currently at epoch %d" % k)
            best_log_posterior, best_z = self.sess.run([self.posterior_components.log_posterior, self.Z], feed_dict)

            for i in range(n):
                resampled_z = self.sess.run(self.Z)

                best_z[:,:,1] = resampled_z[:,:,1]
                feed_dict[self.Z] = best_z
                best_log_posterior += self.sess.run(self.posterior_components.log_posterior,feed_dict)

            best_log_posterior /= n + 1
            ind = np.argmax(best_log_posterior.ravel())

            if best_log_posterior.ravel()[ind] > best_x_planned_posterior:

                print("Plan is updated. Average log posterior for selected plan: %f" % best_log_posterior.ravel()[ind])
                targets = (self.model.phi.S_past_world_frame, self.S_world_frame) + self.posterior_components.inference_targets
                past_states_local, x_planned, log_prior, log_goal_likelihoods, log_posteriors = self.sess.run(targets, feed_dict)
                x_planned = np.reshape(x_planned,(120,2,30,2))

                best_x_planned = x_planned[ind]
                best_x_planned_posterior = best_log_posterior.ravel()[ind]

        # resampled_z = self.sess.run(self.Z)
        #
        # # Get the best robot z
        # actual_best_z_r = best_z[np.unravel_index(ind, best_z.shape[:2])][0]
        #
        # # Copy the best robot z to all of the other z^r slots.
        # best_z[:,:,0] = actual_best_z_r.copy()
        #
        # # Copy new human samples to all of the z^h slots.
        # best_z[:,:,1] = resampled_z[:,:,1]
        #
        # # Set up the feeds.
        # feed_dict[self.Z] = best_z
        #
        # # Compute the rollouts using the best z^r and random z^h
        # contingency = self.sess.run(self.S_world_frame, feed_dict)
        #
        # # Reshape the results
        # contingency = np.reshape(contingency, (120,2,30,2))
        #
        # fig, axes = plt.subplots(10, 10, figsize=(50,50))
        # axes = axes.ravel()
        #
        # for i in range(min(len(axes), 120)):
        #     ax = axes[i]
        #     ax.cla()
        #     ax.axis('off')
        #     ax.set_aspect('equal')
        #     ax.set_xlim(150,220)
        #     ax.set_ylim(-200,-150)
        #     ax.invert_xaxis()
        #
        #     traj = contingency[i]
        #
        #     ego_traj = traj[0]
        #     actor_traj = traj[1]
        #     ego_traj_x = np.array(ego_traj)[:,0].tolist()
        #     ego_traj_y = np.array(ego_traj)[:,1].tolist()
        #     actor_traj_x = np.array(actor_traj)[:,0].tolist()
        #     actor_traj_y = np.array(actor_traj)[:,1].tolist()
        #
        #     ax.plot(ego_traj_x,ego_traj_y,
        #     linestyle = '-',
        #     marker='o',
        #     markersize=3,
        #     zorder=1)
        #
        #     ax.plot(actor_traj_x,actor_traj_y,
        #     linestyle = '-',
        #     marker='o',
        #     markersize=3,
        #     zorder=1)
        #
        # fig.tight_layout(0.0)
        # plt.show()
        # plt.close("all")



        # resampled_z = self.sess.run(self.Z)
        # feed_dict[self.Z] = resampled_z
        # contingency = self.sess.run(self.S_world_frame,feed_dict)
        # contingency = np.reshape(contingency,(120,2,30,2))
        # fig, axes = plt.subplots(10, 10, figsize=(50,50))
        # axes = axes.ravel()
        #
        # for i in range(min(len(axes), 120)):
        #     ax = axes[i]
        #     ax.cla()
        #     ax.axis('off')
        #     ax.set_aspect('equal')
        #     ax.set_xlim(150,220)
        #     ax.set_ylim(-200,-150)
        #     ax.invert_xaxis()
        #
        #     traj = contingency[i]
        #
        #     ego_traj = traj[0]
        #     actor_traj = traj[1]
        #     ego_traj_x = np.array(ego_traj)[:,0].tolist()
        #     ego_traj_y = np.array(ego_traj)[:,1].tolist()
        #     actor_traj_x = np.array(actor_traj)[:,0].tolist()
        #     actor_traj_y = np.array(actor_traj)[:,1].tolist()
        #
        #     ax.plot(ego_traj_x,ego_traj_y,
        #     linestyle = '-',
        #     marker='o',
        #     markersize=3,
        #     zorder=1)
        #
        #     ax.plot(actor_traj_x,actor_traj_y,
        #     linestyle = '-',
        #     marker='o',
        #     markersize=3,
        #     zorder=1)
        #
        # fig.tight_layout(0.0)
        # plt.show()
        # plt.close("all")



        # resampled_z = self.sess.run(self.Z)
        #
        # # Get the best robot z
        # actual_best_z_r = best_z[np.unravel_index(ind, best_z.shape[:2])][0]
        #
        # # Copy the best robot z to all of the other z^r slots.
        # best_z[:,:,0] = actual_best_z_r.copy()
        #
        # # Copy new human samples to all of the z^h slots.
        # best_z[:,:,1] = resampled_z[:,:,1]
        #
        # # Set up the feeds.
        # feed_dict[self.Z] = best_z
        #
        # # Compute the rollouts using the best z^r and random z^h
        # contingency = self.sess.run(self.S_world_frame, feed_dict)
        #
        # # Reshape the results
        # contingency = np.reshape(contingency, (120,2,30,2))
        #
        # for i in range(12):
        #     fig, ax = plt.subplots(1,1, figsize=(50,50))
        #     ax.cla()
        #     ax.set_aspect('equal')
        #     ax.set_xlim(160,230) #150,220
        #     ax.set_ylim(-190,-140) #-200,-150
        #     ax.invert_xaxis()
        #
        #     traj = contingency[i]
        #
        #     ego_traj = traj[0]
        #     actor_traj = traj[1]
        #     ego_traj_x = np.array(ego_traj)[:,0].tolist()
        #     ego_traj_y = np.array(ego_traj)[:,1].tolist()
        #     actor_traj_x = np.array(actor_traj)[:,0].tolist()
        #     actor_traj_y = np.array(actor_traj)[:,1].tolist()
        #
        #     ax.plot(ego_traj_x,ego_traj_y,
        #     linestyle = '-',
        #     lw = 2,
        #     marker='o',
        #     markersize=4,
        #     zorder=1)
        #
        #     ax.plot(actor_traj_x,actor_traj_y,
        #     linestyle = '-',
        #     lw = 2,
        #     marker='o',
        #     markersize=4,
        #     zorder=1)
        #
        #     for i in range(30):
        #         ax.plot([actor_traj_x[i], ego_traj_x[i]],[actor_traj_y[i], ego_traj_y[i]],
        #         linestyle = '-',
        #         lw = 0.5,
        #         marker='o',
        #         markersize=1,
        #         zorder=1)
        #
        #     plt.show()
        #     plt.close("all")

        return best_x_planned,best_x_planned_posterior

class OverconfidentPlanner:
    # optimizing over both the robot and human z
    @classu.member_initialize
    def __init__(self, model, goal_likelihood, sess, check_gradients=False):
        """ Stores the model, constructs the posterior, enables gradient-ascent on the posterior

        :param model:
        :param goal_likelihood:
        :param sess:
        :param check_gradients: whether to check gradients exist.

        """

        log.info("Instantiating {}.".format(self.__class__.__name__))
        # --- Define some useful functions.
        # (B, K, A, T, D) -> (B, K, T, D)
        partial_derivative = lambda t, dv: tf.gradients(t, dv)[0]

        # --- Extract and shape the trajectories and planning variables from the model.
        sampled_output = model.sampled_output
        sampled_rollout = sampled_output.rollout
        self.objective_shape = (10,12)

        # (10, 12, 2, 30, 2)
        self.Z = sampled_output.base_and_log_q.Z_sample
        # (10, 12, 2, 30, 2)
        self.S_world_frame = sampled_rollout.S_world_frame
        # (10,12)
        log_prior = tf.identity(sampled_output.base_and_log_q.log_q_samples, "log_prior")

        # --- Compute the log goal likelihood and form the planning criterion.
        # (10,12)
        goal_log_likelihood_values = tf.identity(goal_likelihood.log_prob(self.S_world_frame), "log_goal_likelihoods")
        # (10,12)
        #log_posteriors = tf.identity(log_prior + goal_log_likelihood_values, "log_posteriors")
        log_posteriors = tf.identity(goal_log_likelihood_values, "log_posteriors") #changed

        # --- Compute the gradients.
        log.info("Computing planning log-posterior gradients...")
        # (10, 12, 2, 30, 2)
        #dlogq_dz = tf.identity(partial_derivative(log_prior, self.Z), "dlogq_dz")
        dlogq_dz = tf.zeros_like(tf.identity(partial_derivative(log_prior, self.Z), "dlogq_dz")) #changed

        if goal_likelihood.describe() == "EmptyGoalLikelihood": dloggoal_dz = tf.zeros_like(dlogq_dz)
        else: dloggoal_dz = tf.identity(partial_derivative(goal_log_likelihood_values, self.Z),"dloggoal_dz")

        # (10, 12, 2, 30, 2)
        #dlog_posterior_dz = tf.identity(dlogq_dz + dloggoal_dz, "dlog_posterior_dz")
        dlog_posterior_dz = tf.identity(dloggoal_dz, "dlog_posterior_dz") #changed

        # -- Package the components of the posterior.
        self.posterior_components = PosteriorComponents(
            log_prior=log_prior,
            log_goal_likelihood=goal_log_likelihood_values,
            log_posterior=log_posteriors,
            dlog_prior_dz=dlogq_dz,
            dlog_goal_likelihood_dz=dloggoal_dz,
            dlog_posterior_dz=dlog_posterior_dz)
        log.info("Done computing gradients.")
        self.posterior_components.validate_shape(self.objective_shape)
        log.info("Done instantiating {}.".format(self.__class__.__name__))

        self.times = 0
        self.saved = None

    def plan(self,feed_dict, n = 100, epochs = 3):

        self.times += 1

        if self.Z in feed_dict:
            del feed_dict[self.Z]

        best_x_planned = None
        best_x_planned_posterior = -float('inf')

        for k in range(epochs):
            log.info("Currently at epoch %d" % k)
            best_log_posterior, best_z = self.sess.run([self.posterior_components.log_posterior, self.Z], feed_dict)

            ind = np.argmax(best_log_posterior.ravel())

            if best_log_posterior.ravel()[ind] > best_x_planned_posterior:

                feed_dict[self.Z] = best_z

                print("Plan is updated. Average log posterior for selected plan: %f" % best_log_posterior.ravel()[ind])
                targets = (self.model.phi.S_past_world_frame, self.S_world_frame) + self.posterior_components.inference_targets
                past_states_local, x_planned, log_prior, log_goal_likelihoods, log_posteriors = self.sess.run(targets, feed_dict)
                x_planned = np.reshape(x_planned,(120,2,30,2))

                best_x_planned = x_planned[ind]
                best_x_planned_posterior = best_log_posterior.ravel()[ind]

        return best_x_planned, best_x_planned_posterior


class UnderconfidentPlanner:
    # optimizing over both the robot and human z
    @classu.member_initialize
    def __init__(self, model, goal_likelihood, sess, all_trajectories):
        """ Stores the model, constructs the posterior, enables gradient-ascent on the posterior

        :param model:
        :param goal_likelihood:
        :param sess:
        :param check_gradients: whether to check gradients exist.

        """

        log.info("Instantiating {}.".format(self.__class__.__name__))
        # --- Define some useful functions.
        # (B, K, A, T, D) -> (B, K, T, D)
        partial_derivative = lambda t, dv: tf.gradients(t, dv)[0]

        # --- Extract and shape the trajectories and planning variables from the model.
        sampled_output = model.sampled_output
        sampled_rollout = sampled_output.rollout
        self.objective_shape = (10,12)

        # (10, 12, 2, 30, 2)
        self.Z = sampled_output.base_and_log_q.Z_sample
        # (10, 12, 2, 30, 2)
        self.S_world_frame = sampled_rollout.S_world_frame
        # (10,12)
        log_prior = tf.identity(sampled_output.base_and_log_q.log_q_samples, "log_prior")

        # --- Compute the log goal likelihood and form the planning criterion.
        # (10,12)
        goal_log_likelihood_values = tf.identity(goal_likelihood.log_prob(self.S_world_frame), "log_goal_likelihoods")
        # (10,12)
        #log_posteriors = tf.identity(log_prior + goal_log_likelihood_values, "log_posteriors")
        log_posteriors = tf.identity(goal_log_likelihood_values, "log_posteriors") #changed

        # --- Compute the gradients.
        log.info("Computing planning log-posterior gradients...")
        # (10, 12, 2, 30, 2)
        #dlogq_dz = tf.identity(partial_derivative(log_prior, self.Z), "dlogq_dz")
        dlogq_dz = tf.zeros_like(tf.identity(partial_derivative(log_prior, self.Z), "dlogq_dz")) #changed

        if goal_likelihood.describe() == "EmptyGoalLikelihood": dloggoal_dz = tf.zeros_like(dlogq_dz)
        else: dloggoal_dz = tf.identity(partial_derivative(goal_log_likelihood_values, self.Z),"dloggoal_dz")

        # (10, 12, 2, 30, 2)
        #dlog_posterior_dz = tf.identity(dlogq_dz + dloggoal_dz, "dlog_posterior_dz")
        dlog_posterior_dz = tf.identity(dloggoal_dz, "dlog_posterior_dz") #changed

        # -- Package the components of the posterior.
        self.posterior_components = PosteriorComponents(
            log_prior=log_prior,
            log_goal_likelihood=goal_log_likelihood_values,
            log_posterior=log_posteriors,
            dlog_prior_dz=dlogq_dz,
            dlog_goal_likelihood_dz=dloggoal_dz,
            dlog_posterior_dz=dlog_posterior_dz)
        log.info("Done computing gradients.")
        self.posterior_components.validate_shape(self.objective_shape)
        log.info("Done instantiating {}.".format(self.__class__.__name__))

        self.decision = None


    def plan(self):

        if self.decision is not None:
            return self.decision

        choice = self.goal_likelihood.constants.choice
        self.goal_likelihood.constants.resample(choice=0)

        underconfident_behavior = self.all_trajectories[4][:,:240:8] # 2, 30, 2
        underconfident_behavior = np.tile(underconfident_behavior[None][None],(10,12,1,1,1))
        underconfident_score = self.sess.run(self.goal_likelihood.log_prob(underconfident_behavior))[0][0]
        scores = []
        for i in range(4):
            behavior = self.all_trajectories[i][:,:240:8] # 2, 30, 2
            behavior = np.tile(behavior[None][None],(10,12,1,1,1))
            score = self.sess.run(self.goal_likelihood.log_prob(behavior))[0][0]
            scores.append(score)
        if underconfident_score > np.mean(scores):
            self.decision = 5
            self.goal_likelihood.constants.resample(choice=choice)
            return self.decision
        print("Error in planning!")
        exit()

class Plan:
    @classu.member_initialize
    def __init__(self,
                 planned_trajectories,
                 past_states_local,
                 z_planned,
                 log_prior,
                 log_posterior,
                 goal_log_likelihoods,
                 useful_plans_mask,
                 planner,
                 steps=0,
                 planned_trajectories_tf=None,
                 valid_plan_indicators=None,
                 best_log_posterior=None):
        """The object that holds plan(s) and their metadata.

        :param planned_trajectories:
        :param past_states_local:
        :param z_planned:
        :param log_prior:
        :param log_posterior:
        :param goal_log_likelihoods:
        :param useful_plans_mask:
        :param planner:
        :param steps:
        :param planned_trajectories_tf:
        :param valid_plan_indicators:
        :param best_log_posterior:
        :returns:
        :rtype:

        """
        self.planned_trajectories = planned_trajectories
        self.past_states_local = past_states_local
        self.planned_trajectories_tf = planned_trajectories_tf
        self._lock()
        assert(z_planned.shape == self.planned_trajectories.shape)
        self.current_target_forward_speed = np.nan
        self.current_forward_speed_error = np.nan
        # The count of the number of plans.
        self.size = z_planned.shape[0] * z_planned.shape[1]
        # The shape of a single plan.
        self.single_plan_shape = z_planned.shape[-3:]
        # Reshape the trajectories to be indexed by flat plan ordering.
        self.planned_trajectories_flat = self.planned_trajectories.reshape((self.size,) + self.single_plan_shape)
        # Compute the ordering of planned trajectories from highest log_posterior to lowest log_posterior.
        self.plan_ordering_flat = np.argsort(log_posterior.ravel())[::-1]
        # Store the best plan.
        self.best_planned_traj = self.planned_trajectories_flat[self.plan_ordering_flat[0]]
        # Store the flat mask.
        self.useful_plans_mask_flat = useful_plans_mask.ravel()
        # Count the number of useful plans.
        self.n_useful = self.useful_plans_mask_flat.sum()
        # If the best plan's not useful, it's risky.
        self.is_risky = not self.useful_plans_mask_flat[self.plan_ordering_flat[0]]
        # Store values of the best plan.
        self.best_plan_log_prior = self.log_prior.ravel()[self.plan_ordering_flat[0]]
        self.best_plan_log_posterior = self.log_posterior.ravel()[self.plan_ordering_flat[0]]
        self.best_plan_loggoal = self.goal_log_likelihoods.ravel()[self.plan_ordering_flat[0]]

        # (2,) ?
        self.best_planned_traj_goal = self.best_planned_traj[..., -1, :]

    def _lock(self):
        # Lock important data.
        npu.lock_nd(self.log_posterior)
        npu.lock_nd(self.z_planned)
        npu.lock_nd(self.log_prior)
        npu.lock_nd(self.useful_plans_mask)
        npu.lock_nd(self.past_states_local)
        npu.lock_nd(self.planned_trajectories)

    def get_plan_position(self, t_future_index):
        """Returns the t'th position into the planned trajectories (in coordinates of original frame)

        :param t_future_index:
        :returns:
        :rtype:

        """
        # Get the first trajectory from the ordering.
        # Get the t'th index of the trajectory.
        return self.best_planned_traj[t_future_index, :]
