import gym
import os
import sys
import copy 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gym import wrappers
from datetime import datetime

import pdb

# turns list of integers into an int
# Ex.
# build_state([1, 2, 3, 4, 5]) -> 12345
def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))
    # map(function, iterable(list, ...))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

class FeatureTransformer: # transforms state Box object to sequential integers
    def __init__(self):
        # Note: to make this better you could look at how often each bin was
        # actually used while running the script.
        # It's not clear from the high/low values nor sample() what values
        # we really expect to get.
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self, observation):
        # returns an int
        # pdb.set_trace()
        cart_pos = observation[0]
        cart_vel = observation[1]
        pole_angle = observation[2]
        pole_vel = observation[3]
        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins)
        ])

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        # check env.observation_space.shape[0]
        # check env.action_space.n
        # pdb.set_trace()
        num_states = 10**env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions)) # Q values

    def predict(self, s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]

    def update(self, s, a, G):
        # pdb.set_trace()
        x = self.feature_transformer.transform(s)
        self.Q[x,a] += 1e-2*(G - self.Q[x,a]) # step_size = 1e-2

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample() # explore action selection

        else:
            p = self.predict(s)
            return np.argmax(p) # greedy action selection

def play_one(model, eps, gamma): # gamma: discount factor
    # play only one episode!
    observation = env.reset()[0]
    terminated = False
    totalreward = 0
    iters = 0
    while not terminated and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)
        totalreward += reward

        if terminated and iters < 199:
            reward = -300

        next_action = model.sample_action(observation, eps) # next action using same policy (on policy)
        next_observation = ft.transform(observation)

        G = reward + gamma*model.Q[next_observation, next_action] # SARSA uses next policy to calculate gain (on-policy)
        # G = reward + gamma*np.max(model.predict(observation)) # since it is q_learning uses np.max **not using policy** (off-policy)
        # pdb.set_trace()
        model.update(prev_observation, action, G)

        iters += 1

    return totalreward

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean() # average of 100 totalrewards

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9 # discount factor

    # if 'monitor' in sys.argv:
    #     print("basename of file: ",os.path.basename(__file__))
    #     filename = os.path.basename(__file__).split('.')[0]
    #     monitor_dir = './' + filename + '_' + str(datetime.now())
    #     env = wrappers.Monitor(env, monitor_dir)
    
    N = 10000
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1) # exploration term reduces as iterations.
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
    
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
