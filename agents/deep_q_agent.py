import memory
import gym
import numpy as np
import nn


class DeepQAgent(object):

    def __init__(self, hidden_layers, discount=1.0, memory_size=10000):
        self.hidden_layers = hidden_layers
        self.discount = discount
        self.memory = memory.Memory(memory_size)

    def register_env(self, env):
        self.Xdim = env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            self.Ydim = env.action_space.n
        network = nn.NeuralNetwork(self.Xdim, self.Ydim, self.hidden_layers)
        self.q = nn.NeuralAgent(network, self.discount)

    def train_epoch(self, env, target_reward=1000, learn=True):
        eps = .1
        done = False
        cum_reward = 0
        state = env.reset()
        while not done and cum_reward < target_reward:
            if np.random.uniform() < eps:
                action = np.random.randint(0, self.Ydim)
            else:
                action = self.q.action([state])
            next_state, reward, done, _ = env.step(action)

            self.memory.append(state, action, reward, next_state, done)
            if learn:
                self.learn()
            cum_reward += reward
            state = next_state
        return cum_reward

    def learn(self):
        state, rewards, next_state, terminals = self.memory.minibatch(1000)
        self.q.trainstep(state, rewards, next_state, terminals)

    def loss(self):
        state, rewards, next_state, terminals = self.memory.minibatch(1000)
        return np.sqrt(self.q.calcloss(state, rewards, next_state, terminals)) / 1000

    def sample(self):
        state, rewards, next_state, terminals = self.memory.minibatch(10)
        storedq = self.q.storedq(state)
        print np.mean(storedq)

    def new_epoch(self):
        pass
