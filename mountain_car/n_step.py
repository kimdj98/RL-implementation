import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

from gym import wrappers
from datetime import datetime

import q_learning
from q_learning import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg

class SGDRegressor:
    def __init__(self, **kwargs):
        self.w =  None
        self.lr = 1e-2

    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)

        self.w += self.lr * (Y-X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

# replace SKLearn Regressor
q_learning.SGDRegressor = SGDRegressor

# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, n=5): # play one episode
    observation = env.reset()[0]
    done = False
    totalreward = 0
    rewards = []
    states = []
    actions = []
    iters = 0 
    # array of [gamma^0, gamma^1, ..., gamma^(n-1)]
    multiplier = np.array([gamma]*n)**np.arange(n)
    # while not done and iters < 10000:
    while not done and iters < 10000:
        action = model.sample_action(observation, eps) # policy: epsilon greedy

        states.append(observation)
        actions.append(action)

        prev_observation = observation
        observation, reward, done, truncated, info =env.step(action)

        rewards.append(reward)

        # update the model

        if len(rewards) >= n: # update occurs after n-step!
            # return_up_to_prediction = calculate_reutrn_before_prediction(rewards, gamma)
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            # pdb.set_trace() # check model.predict(observation[0])
            G = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation))
            model.update(states[-n], actions[-n], G)

        # Q) why do we need this? -> to speed up but cannot draw 
        # if len(rewards) > n:
        #     rewards.pop(0)
        #     states.pop(0)
        #     actions.pop(0)
        # assert(len(rewards) <= n)

        totalreward += reward
        iters += 1

    # empty the cache
    if n == 1:
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n+1:]
        states = states[-n+1:]
        actions = actions[-n+1:]

    if observation[0] >= 0.5:
        # print("Episode_ends.")
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    return totalreward

if __name__=="__main__":
    env = gym.make("MountainCar-v0")
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99 # discount factor

    # training
    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        
        print("episode:", n+1, "total rewards:", totalreward, "eps:", eps)
    print("avg rewards for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)