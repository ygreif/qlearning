import gym
import random
import math
from agents import deep_q_agent
import tensorflow as tf


class Parameters(object):

    def __init__(self, learning_rate=0.001, learner='adam', hidden_layers=[20], memory_size=10000, minibatch_size=100, eps=.1, nonlinearity=tf.nn.tanh, init='normal', init_bias=0.0, use_prob=False):
        self.learning_rate = learning_rate
        self.learner = learner
        self.hidden_layers = hidden_layers
        self.memory_size = memory_size
        self.minibatch_size = int(minibatch_size)
        self.eps = eps
        self.nonlinearity = nonlinearity
        self.init = init
        self.init_bias = float(init_bias)
        self.use_prob = use_prob

    def __str__(self):
        return "learning_rate={}, learner={}, hidden_layers={}, memory_size={}, minibatch={}, eps={}, nonlinearity={}, init={}, init_bias={}, use_prob={}".format(str(self.learning_rate), str(self.learner), ','.join([str(layer) for layer in self.hidden_layers]), str(self.memory_size), str(self.minibatch_size), str(self.eps), str(self.nonlinearity), self.init, str(self.init_bias), str(self.use_prob))


class Trainer(object):

    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def gen_parameters(self):
        hidden_layers = []
        for _ in range(random.randint(2, 5)):
            hidden_layers.append(random.randint(40, 300))
        lear
        ner = 'adam'
        nonlinearity = random.choice([tf.nn.softplus, tf.nn.tanh, tf.nn.tanh])
        init = random.choice(['normal', 'normal', 'uniform'])

        learning_rate = math.pow(10, -1 * random.randint(3, 5))
        minibatch_size = math.pow(10, random.randint(1, 3))

        if random.random() < .5:
            use_prob = False
            init_bias = random.choice([0.0, 1.0])
            eps = random.random() / 5.0
        else:
            use_prob = True
            eps = random.random()
            init_bias = random.choice([1.0, 1.0, 10.0])

        return Parameters(
            learning_rate=learning_rate, learner=learner, hidden_layers=hidden_layers, minibatch_size=minibatch_size, eps=eps, nonlinearity=nonlinearity, init=init, init_bias=init_bias, use_prob=use_prob)

    def train(self, target_reward, max_epochs=200, parameters=Parameters(), out=True):
        self.agent = deep_q_agent.DeepQAgent(self.env, parameters)

        for epoch in range(max_epochs):
            reward = self.agent.train_epoch(self.env, target_reward)
            if epoch % 250 == 0:
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
