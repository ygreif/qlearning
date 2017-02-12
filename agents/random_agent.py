import gym
import numpy as np


class RandomAgent(object):

    def __init__(self):
        self.weights = None

    def new_epoch(self):
        pass

    def reshape(self, env):
        X = env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            self.action = self.discrete_action
            Y = env.action_space.n
        elif type(env.action_space) == gym.spaces.box.Box:
            self.action = self.continuous_action
            Y = env.action_space.shape[0]
        self.weights = np.random.standard_normal((X, Y))

    def train_epoch(self, env, target_reward=1000):
        self.weights = np.random.standard_normal(self.weights.shape)

        done = False
        cum_reward = 0
        state = env.reset()
        self.new_epoch()
        while not done and cum_reward < target_reward:
            action = self.action(state)
            state, reward, done, _ = env.step(action)
            cum_reward += reward
        return cum_reward

    def discrete_action(self, state):
        return np.argmax(np.dot(state, self.weights))

    def continuous_action(self, state):
        return np.dot(state, self.weights)
