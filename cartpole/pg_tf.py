import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import gym
from q_learning import plot_running_avg

# approximates pi(a | s)
class PolicyModel:
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        inp = tf.keras.layers.Input(input_size)
        self.output_size = output_size
        net = inp
        for hidden_layer_size in hidden_layer_sizes:
            net = tf.keras.layers.Dense(hidden_layer_size, activation='tanh')(net)
        net = tf.keras.layers.Dense(output_size, activation='softmax', use_bias=False)(net)

        self.model = tf.keras.models.Model(inp, net)
        self.model.build(input_size)
 
        self.optimizer = tf.keras.optimizers.Adagrad(10e-2)
        self.do_minimize = tf.function(self.minimize,
                                       input_signature=[tf.TensorSpec(shape=(None, input_size), dtype=tf.float32),
                                                        tf.TensorSpec(shape=None, dtype=tf.int32),
                                                        tf.TensorSpec(shape=None, dtype=tf.float32)])
 
    @tf.function
    def calculate_forward(self, input_data):
        return self.model(input_data)
 
    def minimize(self, input_data, actions, advantages):
        def calc_cost():
            p_a_given_s = self.model(input_data)
            selected_probs = tf.math.log(tf.reduce_sum(p_a_given_s * tf.one_hot(actions, self.output_size), 1))
            return -tf.reduce_sum(advantages * selected_probs)
        self.optimizer.minimize(calc_cost, self.model.trainable_variables)
 
    def partial_fit(self, input_data, actions, advantages):
        input_data = np.atleast_2d(input_data)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.do_minimize(input_data, actions, advantages)
 
    def predict(self, input_data):
        input_data = np.atleast_2d(input_data)
        return self.calculate_forward(input_data).numpy()
 
    def sample_action(self, input_data):
        p = self.predict(input_data)[0]
        return np.random.choice(len(p), p=p)
 
# approximates V(s)
class ValueModel:
    def __init__(self, input_size: int, hidden_layer_sizes):
        inp = tf.keras.layers.Input(input_size)
        net = inp
        for hidden_layer_size in hidden_layer_sizes:
            net = tf.keras.layers.Dense(hidden_layer_size, activation='tanh')(net)
        net = tf.keras.layers.Dense(1)(net)
 
        self.model = tf.keras.models.Model(inp, net)
        self.model.build(input_size)
 
        self.optimizer = tf.keras.optimizers.SGD(10e-5)
        self.do_minimize = tf.function(self.minimize,
                                       input_signature=[tf.TensorSpec(shape=(None, input_size), dtype=tf.float32),
                                                        tf.TensorSpec(shape=None, dtype=tf.float32)])
 
    def minimize(self, input_data, target_data):
        def calc_cost():
            y_hat = tf.reshape(self.model(input_data), [-1])
            return tf.reduce_sum(tf.math.square(target_data - y_hat))
        self.optimizer.minimize(calc_cost, self.model.trainable_variables)
 
    def partial_fit(self, input_data, target_data):
        input_data = np.atleast_2d(input_data)
        target_data = np.atleast_1d(target_data)
        self.do_minimize(input_data, target_data)
 
    @tf.function
    def calculate_forward(self, input_data):
        return self.model(input_data)
 
    def predict(self, input_data):
        input_data = np.atleast_2d(input_data)
        return self.calculate_forward(input_data).numpy()

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
        V_next = vmodel.predict(observation)[0]
        G = reward + gamma*V_next

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
        advantages.append(G - vmodel.predict(s)[0])
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