import memory
import numpy as np


class DeltaStrategy(object):

    def __init__(self, epoch, scale, decay, max_steps=250):
        self.scale = max(scale * (decay - epoch) / float(decay), 0.0)
        self.max_steps = max_steps
        self.target = 1000

    def explore(self, action):
        return action + np.random.normal(scale=self.scale, size=len(action))


class Agent(object):

    def __init__(self, q, memory_size=1000000, minibatch_size=100):
        self.q = q
        self.memory = memory.Memory(memory_size)
        self.minibatch_size = minibatch_size

    def train_epoch(self, env, strat):
        explore = strat.explore
        target = strat.target
        max_steps = strat.max_steps

        done = False
        cum_reward = False
        steps = 0
        state = env.reset()
        while not done and cum_reward < target and steps < max_steps:
            action = explore(self.action(state))
            action = min(
                max(action, env.action_space.low), env.action_space.high)
            if np.isnan(action):
                return -99999999
            next_state, reward, done, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            self.learn()
            cum_reward += reward
            state = next_state
            steps += 1
#            env.render()
        return cum_reward

    def action(self, state):
        return self.q.action([state])

    def learn(self):
        state, actions, rewards, next_state, terminals = self.memory.minibatch(
            self.minibatch_size)
        self.q.trainstep(state, actions, rewards, next_state, terminals)
