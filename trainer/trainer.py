import gym
import random
import math
from agents import deep_q_agent
import tensorflow as tf


# learning_rate=1e-06, learner=adam, hidden_layers=245,297,218, memory_size=10000, minibatch=1000, eps=0.111775048017, nonlinearity=<function tanh at 0x7fe5e97756e0>, init=normal, init_bias=0.0, use_prob=False
# learning_rate=0.0001, learner=adam, hidden_layers=238,108,285,273,
# memory_size=10000, minibatch=1000, eps=0.543402880186,
# nonlinearity=<function tanh at 0x7fe5e97756e0>, init=uniform,
# init_bias=1.0, use_prob=True

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
        hidden_layers = random.choice((
            [200, 200, 200], [100, 100, 100], [150, 150]))
        use_prob = False
        learner = 'adam'
        nonlinearity = tf.nn.tanh
        init = random.choice(['normal', 'normal'])
        init_bias = 0.0
        learning_rate = learning_rate = random.choice(
            [.001, .0005, .002, .005])
        minibatch_size = random.choice([500, 1000, 1500, 2000])
        memory_size = random.choice([10000, 5000, 20000])
        eps = random.random() / 5.0

        return Parameters(
            learning_rate=learning_rate, learner=learner, hidden_layers=hidden_layers, minibatch_size=minibatch_size, eps=eps, nonlinearity=nonlinearity, init=init, init_bias=init_bias, use_prob=use_prob)

    def train(self, target_reward, max_epochs=200, parameters=Parameters(), out=True):
        self.agent = deep_q_agent.DeepQAgent(self.env, parameters)

        for epoch in range(max_epochs):
            reward = self.agent.train_epoch(self.env, target_reward)
            if epoch % 100 == 0:
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
