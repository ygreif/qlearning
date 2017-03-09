import memory
import numpy as np


class DeltaStrategy(object):

    def __init__(self, epoch, scale, decay, max_steps=1000, target=1000):
        self.scale = max(scale * (decay - epoch) / float(decay), 0.0)
        self.max_steps = max_steps
        self.target = target

    def explore(self, action):
        return action + np.random.normal(scale=self.scale, size=len(action))


class NoExploration(object):

    def __init__(self, max_steps=11000, target=11000):
        self.max_steps = max_steps
        self.target = target

    def explore(self, action):
        return action


class Agent(object):

    def __init__(self, q, memory_size=100000000, minibatch_size=300):
        self.q = q
        self.memory = memory.Memory(memory_size)
        self.minibatch_size = minibatch_size

    def train_epoch(self, env, strat, learn=True):
        explore = strat.explore
        target = strat.target
        max_steps = strat.max_steps

        done = False
        cum_reward = False
        steps = 0
        state = env.reset()
        while not done and cum_reward < target and steps < max_steps:
            action = explore(self.action(state))
            action = np.minimum(
                np.maximum(action, env.action_space.low), env.action_space.high)
            if np.isnan(action).any():
                return -99999999
            next_state, reward, done, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            if learn:
                self.learn()
            cum_reward += reward
            state = next_state
            steps += 1
            env.render()
            if steps > 11000 and steps % 100 == 0:
                print "On step", steps
        return cum_reward

    def action(self, state):
        return self.q.action([state])

    def learn(self):
        minibatch = self.memory.minibatch(self.minibatch_size)
        self.q.trainstep(minibatch.state, minibatch.actions,
                         minibatch.rewards, minibatch.next_state, minibatch.terminals)
