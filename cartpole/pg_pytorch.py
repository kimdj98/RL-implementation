import os
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
import numpy as np
import matplotlib.pyplot as plt

import gym
from q_learning import plot_running_avg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# approximates pi(a | s)
class PolicyModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes=[]):
        super(PolicyModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        modules = []

        sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(len(sizes)-1):
            if i != len(sizes) -2:
                modules.append(nn.Linear(sizes[i], sizes[i+1]))
                modules.append(nn.Softplus()) # activation function
            else:
                modules.append(nn.Linear(sizes[i], sizes[i+1]))
                modules.append(nn.Softmax(dim=1))

        for module in modules:
            if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_normal_(module.weight.data)
                    torch.nn.init.zeros_(module.bias.data)
        
        self.model = nn.Sequential(*modules)

        # hyperparameter
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def loss_fn(self, input_data, actions, advantages):
        input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = torch.atleast_2d(input_data)
        actions = torch.tensor(actions)
        actions = torch.atleast_2d(actions)
        actions = actions.type('torch.LongTensor') # change actions datatype to int for one_hot encoding
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = torch.atleast_2d(advantages)
        p_a_given_s = self.model(input_data)
        selected_probs = torch.log(torch.sum(p_a_given_s * F.one_hot(actions, self.output_size), dim=2))
        # pdb.set_trace()
        loss = -torch.sum(advantages * selected_probs)
        # print(f"{loss=}")
        return loss

    def partial_fit(self, input_data, actions, advantages):
        input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = torch.atleast_2d(input_data)
        actions = torch.tensor(actions, dtype=torch.float32)
        actions = torch.atleast_2d(actions)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = torch.atleast_2d(advantages)

        # forward path
        loss = self.loss_fn(input_data, actions, advantages)

        # backward path
        self.optimizer.zero_grad()
        loss.backward()

        #update
        # print(loss)
        self.optimizer.step()
        # print(self.loss_fn(input_data, actions, advantages))
        # pdb.set_trace()

    def predict(self, input_data):
        input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = torch.atleast_2d(input_data)
        return self.model(input_data)

    def forward(self, input_data):
        input_data = torch.tensor(input_data, dtype=torch.float32)
        return self.predict(input_data)

    def sample_action(self, input_data):
        input_data = torch.tensor(input_data, dtype=torch.float32)
        p = self.predict(input_data)[0]
        # pdb.set_trace()
        # print(p)
        # print(input_data)
        return torch.multinomial(p, num_samples=1).item()

# approximates V(s)
class ValueModel(nn.Module):
    def __init__(self, input_size: int, hidden_layer_sizes):
        super(ValueModel, self).__init__()
        self.input_size = input_size
        self.output_size = 1
        modules = []

        sizes = [input_size] + hidden_layer_sizes + [self.output_size]
        for i in range(len(sizes)-1):
            if i != len(sizes) - 2:
                modules.append(nn.Linear(sizes[i], sizes[i+1]))
                modules.append(nn.Softplus()) # activation function
            else:
                modules.append(nn.Linear(sizes[i], sizes[i+1]))

        for module in modules:
            if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_normal_(module.weight.data)
                    torch.nn.init.zeros_(module.bias.data)

        self.model = nn.Sequential(*modules)
        
        # hyperparameter
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def loss_fn(self, input_data, target_data):
        input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = torch.atleast_2d(input_data)
        target_data = torch.tensor(target_data, dtype=torch.float32)
        target_data = torch.atleast_1d(target_data)

        # pdb.set_trace()
        loss = (target_data - self.model(input_data)).square().sum()
        # print(f"{loss=}")
        return loss
        # return torch.sum(torch.square(target_data - input_data))

    def partial_fit(self, input_data, target_data):
        input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = torch.atleast_2d(input_data)
        target_data = torch.tensor(target_data, dtype=torch.float32)
        target_data = torch.atleast_1d(target_data)
        
        # forward pass
        loss = self.loss_fn(input_data, target_data)

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()

        #update
        # print(loss)
        self.optimizer.step()
        # print(self.loss_fn(input_data, target_data))
        # pdb.set_trace()

    def predict(self, input_data):
        input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = torch.atleast_2d(input_data)
        # pdb.set_trace()
        return self.model(input_data).item()

    def forward(self, input_data):
        return self.predict(input_data)

def play_one_td(env, pmodel, vmodel, gamma):
    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)

        if  done:
            reward = -200

        # update the models
        V_next = vmodel.predict(observation)
        G = reward + gamma * V_next

        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)

        if reward == 1: # if we changed the reward to -200
            totalreward += reward

        iters += 1

    return totalreward

def play_one_mc(env, pmodel, vmodel, gamma):
    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0

    states = []
    actions = []
    rewards = []

    reward = 0
    
    while not done and iters < 2000:
        action = pmodel.sample_action(observation)

        states.append(observation)
        actions.append(action)
        rewards.append(reward)

        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)

        if done:
            reward = -200

        if reward == 1:
            totalreward += reward

        iters += 1

    action = pmodel.sample_action(observation)
    states.append(observation)
    actions.append(action)
    rewards.append(reward)

    returns = []
    advantages = []
    G = 0

    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G - vmodel.predict(s))
        G = r + gamma * G
    returns.reverse()
    advantages.reverse()

    # update the models
    pmodel.partial_fit(states, actions, advantages)
    vmodel.partial_fit(states, returns)

    return totalreward

def main():
    env = gym.make('CartPole-v1')
    D = env.observation_space.shape[0] # dimension of state
    K = env.action_space.n # number of action space
    pmodel = PolicyModel(D, K, [])
    vmodel = ValueModel(D, [10])
    gamma = 0.99

    N = 1000
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        totalreward = play_one_mc(env, pmodel, vmodel, gamma)
        totalrewards[n] = totalreward
        if (n+1) % 100 == 0:
            print("episode:", n+1, "total reward:", totalreward, "avg reward (last 100):", totalrewards[max(0, n-100): n+1].mean())    

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

if __name__=="__main__":
    main()
    pass