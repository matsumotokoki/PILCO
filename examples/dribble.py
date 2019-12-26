import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
import gpflow
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
from myutils import rollout, policy
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
np.random.seed(0)

# Introduces a simple wrapper for the gym environment
# Reduces dimensions, avoids non-smooth parts of the state space that we can't model
# Uses a different number of timesteps for planning and testing
# Introduces priors

class dribbleEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, '/Users/koki/pyfiles/PILCO/examples/world3.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        if type(action) == type(np.ndarray([1])) and len(action) == 2:
            action = 4
        action = int(((action//3)-1)*200), int(((action%3)-1)*200)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        # x, _, y = self.sim.data.body_xpos[0]
        # dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        # v1, v2 = self.sim.data.qvel[1:3]
        # vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        # alive_bonus = 10
        # r = alive_bonus - dist_penalty - vel_penalty
        goal_norm = math.sqrt((90 - ob[2])**2 + (0 - ob[3])**2)
        goal_arr_vec = [(-ob[2]+90)/goal_norm,(-ob[3] + 0)/goal_norm]
        boal_vel = [self.data.qvel[2],self.data.qvel[3]]
        r = goal_arr_vec[0]*boal_vel[0] + goal_arr_vec[0]*boal_vel[1]
        # print(r)
        # done = bool(y <= 1)
        done = False
        r = 10
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            # self.sim.data.qpos[:1],  # cart x pos
            # np.sin(self.sim.data.qpos[1:]),  # link angles
            # np.cos(self.sim.data.qpos[1:]),
            # np.clip(self.sim.data.qvel, -10, 10),
            # np.clip(self.sim.data.qfrc_constraint, -10, 10)
            self.data.body_xpos[1][0:1],
            self.data.body_xpos[1][1:2],
            self.data.body_xpos[2][0:1],
            self.data.body_xpos[2][1:2],
            self.data.qvel[0:1],
            self.data.qvel[1:2],
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        # v.cam.trackbodyid = 0
        # v.cam.distance = self.model.stat.extent * 0.5
        # v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]


class DribbleWarpper():
    def __init__(self):
        # self.env = gym.make('InvertedDoublePendulum-v2').env 
        self.env = dribbleEnv()
        self.action_space = 9
        self.observation_space = 6

    # def state_trans(self, s):
    #     a1 = np.arctan2(s[1], s[3])
    #     a2 = np.arctan2(s[2], s[4])
    #     s_new = np.hstack([s[0], a1, a2, s[5:-3]])
    #     return s_new

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        # if np.abs(ob[0])> 0.98 or np.abs(ob[-3]) > 0.1 or  np.abs(ob[-2]) > 0.1 or np.abs(ob[-1]) > 0.1:
        if ob[2] > 80 and -25 < ob[3] < 25:
            done = True
        # return self.state_trans(ob), r, done, {}
        return ob, r, done, {}

    def reset(self):
        ob =  self.env.reset()
        # return self.state_trans(ob)
        return ob

    def render(self):
        self.env.render()


SUBS = 1
bf = 40
maxiter=80
state_dim = 6
control_dim = 1
max_action=9.0 # actions for these environments are discrete
target = np.zeros(state_dim)
weights = 3.0 * np.eye(state_dim)
weights[0,0] = 0.5
weights[3,3] = 0.5
m_init = np.zeros(state_dim)[None, :]
S_init = 0.01 * np.eye(state_dim)
T = 40
J = 1
N = 12
T_sim = 130
restarts=True
lens = []

with tf.Session() as sess:
    env = DribbleWarpper()

    # Initial random rollouts to generate a dataset
    X,Y = rollout(env, None, timesteps=T, random=True, SUBS=SUBS)
    for i in range(1,J):
        X_, Y_ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim

    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

    # for numerical stability
    for model in pilco.mgpr.models:
        # model.kern.lengthscales.prior = gpflow.priors.Gamma(1,10) priors have to be included before
        # model.kern.variance.prior = gpflow.priors.Gamma(1.5,2)    before the model gets compiled
        model.likelihood.variance = 0.001
        model.likelihood.variance.trainable = False

    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)

        X_new, Y_new = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        # cur_rew = 0
        # for t in range(0,len(X_new)):
        #     cur_rew += reward_wrapper(R, X_new[t, 0:state_dim, None].transpose(), 0.0001 * np.eye(state_dim))[0]
        # print('On this episode reward was ', cur_rew)

        # Update dataset
        X = np.vstack((X, X_new[:T, :])); Y = np.vstack((Y, Y_new[:T, :]))
        pilco.mgpr.set_XY(X, Y)

        lens.append(len(X_new))
        print(len(X_new))
        if len(X_new) > 600: break
