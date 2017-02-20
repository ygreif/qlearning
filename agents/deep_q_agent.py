import memory
import gym
import numpy as np
import nn


class DeepQAgent(object):

    def __init__(self, env, parameters, discount=.99):
        self.discount = discount

        self.memory = memory.Memory(parameters.memory_size)
        self.register_env(env)
        self.create_nn(parameters)
        self.minibatch_size = parameters.minibatch_size
        self.eps = parameters.eps
        self.use_prob = parameters.use_prob

    def create_nn(self, parameters):
        if self.discount:
            network = nn.NeuralNetwork(
                self.Xdim, self.Ydim, parameters.hidden_layers, parameters.nonlinearity, parameters.init, parameters.init_bias)
            self.q = nn.QApproximation(network, parameters, self.discount)
        else:
            network = nn.NeuralNetwork(
                self.Xdim, parameters.hidden_layers[-1], parameters.hidden_layers[:-1], parameters.nonlinearity, parameters.init, parameters.init_bias)
            self.q = nn.NAFApproximation(nn, self.Ydim)

    def register_env(self, env):
        self.Xdim = env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            self.Ydim = env.action_space.n
            self.discrete = True
        elif type(env.action_space) == gym.spaces.box.Box:
            self.discount = False
            self.Ydim = len(env.action_space.low)
            self.low = env.action_space.low
            self.high = env.action_space.high

    def train_epoch(self, env, target_reward=1000, learn=True):
        eps = self.eps
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

    def action(self, state):
        return self.q.action([state])

    def learn(self):
        state, action, rewards, next_state, terminals = self.memory.minibatch(
            self.minibatch_size)
        self.q.trainstep(state, action, rewards, next_state, terminals)

    def loss(self):
        state, rewards, next_state, terminals = self.memory.minibatch(1000)
        return np.sqrt(self.q.calcloss(state, rewards, next_state, terminals)) / 1000

    def sample(self):
        state, rewards, next_state, terminals = self.memory.minibatch(10)
        storedq = self.q.storedq(state)
        print np.mean(storedq)

    def new_epoch(self):
        pass
