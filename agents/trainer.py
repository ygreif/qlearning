import matplotlib.pyplot as plt

import gym
import random
import math
import deep_q_agent


class Parameters(object):

    def __init__(self, learning_rate=0.001, learner='adam', hidden_layers=[20], memory_size=10000, minibatch_size=100):
        self.learning_rate = learning_rate
        self.learner = learner
        self.hidden_layers = hidden_layers
        self.memory_size = memory_size
        self.minibatch_size = int(minibatch_size)

    def __str__(self):
        return "learning_rate={}, learner={}, hidden_layers={}, memory_size={}, minibatch={}".format(str(self.learning_rate), str(self.learner), ','.join([str(layer) for layer in self.hidden_layers]), str(self.memory_size), str(self.minibatch_size))


class Trainer(object):

    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def gen_parameters(self):
        hidden_layers = []
        for _ in range(random.randint(1, 3)):
            hidden_layers.append(random.randint(5, 200))
        if random.random() < .5:
            learner = 'adam'
        else:
            learner = 'gradient'
        learning_rate = math.pow(10, -1 * random.randint(2, 6))
        minibatch_size = math.pow(10, random.randint(1, 3))

        return Parameters(
            learning_rate=learning_rate, learner=learner, hidden_layers=hidden_layers, minibatch_size=minibatch_size)

    def train(self, target_reward, max_epochs=200, parameters=Parameters(), out=True):
        self.agent = deep_q_agent.DeepQAgent(self.env, parameters)

        '''
        for _ in range(20):
            reward = self.agent.train_epoch(
                self.env, target_reward, learn=False)
            print reward
        if out:
            print "Train"
        for _ in range(1000):
            self.agent.learn()
            self.agent.sample()
        if out:
            print "Now for the main event"
        '''
        for epoch in range(max_epochs):
            reward = self.agent.train_epoch(self.env, target_reward)
#            for _ in range(10):
#                self.agent.learn()
            if epoch % 50 == 0:
                loss = self.agent.loss()
                if out:
                    print "Epoch {} Reward {} Loss {}".format(epoch, reward, loss)
            if reward >= target_reward:
                break

        return {'reward': reward, 'epoch': epoch}

    def run(self, max_steps=1000):
        state = self.env.reset()
        step = 0
        while True:
            step += 1
            self.env.render()
            state, reward, done, _ = self.env.step(self.agent.action(state))
            if done or (max_steps and step >= max_steps):
                break
        print "Episode finished after {} timesteps".format(step)

if __name__ == '__main__':
    trainer = Trainer('CartPole-v0')
    trainer.search_parameters()
#    trainer.train(200, 2000)
