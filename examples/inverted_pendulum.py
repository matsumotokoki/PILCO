import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
np.random.seed(0)

# env = gym.make('InvertedPendulum-v2')
# env = gym.make('MountainCarContinuous-v0')
env = gym.make('Pendulum-v0')

SUBS=2

def rollout(policy, timesteps, verbose=False):
    X = []; Y = []
    env.reset()
    x, _, _, _ = env.step([0])
    for timestep in range(timesteps):
        env.render()
        u = policy(x)
        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
        if verbose:
            print("Action: ", u)
            print("State : ",  x_new)
        if done: break
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)

def random_policy(x):
    return env.action_space.sample()

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]

# Initial random rollouts to generate a dataset
X,Y = rollout(policy=random_policy, timesteps=40)
for i in range(1,3):
    X_, Y_ = rollout(policy=random_policy, timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))


state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5, max_action=2.0)
# controller = LinearController(state_dim=state_dim, control_dim=control_dim, max_action=2.0)

# Example of user provided reward function, setting a custom target state
# R = ExponentialReward(state_dim=state_dim, t=np.array([0.45, 0.0])) # mountain car
R = ExponentialReward(state_dim=state_dim, t=np.array([1.0,0,0]), W=np.diag([2.0,2.0,0.0001]))

pilco = PILCO(X, Y, controller=controller, horizon=50, reward=R, m_init=[-1.0,0,0])

# Example of fixing a parameter, optional, for a linear controller only
#pilco.controller.b = np.array([[0.0]])
#pilco.controller.b.trainable = False

for rollouts in range(5):
    pilco.optimize()
    # import pdb; pdb.set_trace()
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=50, verbose=True)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)
    if rollouts<2:
        controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5, max_action=2.0)
        # controller = LinearController(state_dim=state_dim, control_dim=control_dim)
        pilco = PILCO(X, Y, controller=controller, horizon=50, reward=R, m_init=[-1.0,0,0])
