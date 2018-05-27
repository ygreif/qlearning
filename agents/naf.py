import gym
import gym.spaces.box
import tensorflow as tf
import numpy as np

import nn


class SetupNAF(object):

    @classmethod
    def setup(cls, env, nnvParameters, nnpParameters, nnqParameters, learningParameters):
        assert type(env.action_space) == gym.spaces.box.Box
        indim = env.observation_space.shape[0]
        actiondim = len(env.action_space.low)
        low = env.action_space.low
        high = env.action_space.high
        print low, high
        nnv = nn.NeuralNetwork(indim, 1, **nnvParameters)
        if actiondim == 1:
            pdim = 1
        else:
            pdim = (actiondim) * (actiondim + 1) / 2
        nnp = nn.NeuralNetwork(indim, pdim, **nnpParameters)
        nnq = nn.NeuralNetwork(indim, actiondim, **nnqParameters)
        naf = NAFApproximation(
            nnv, nnp, nnq, low, high, actiondim, **learningParameters)

        # initialize NAF so actions are in range
        n = 1000
        success = False
        obs_space_small = gym.spaces.box.Box(-100,
                                             100, env.observation_space.shape)
        print obs_space_small.shape
        print env.observation_space.shape
        target_action = low + (high - low) / 2.0
        targets = [target_action for i in range(n)]
        for _ in range(1000):
            # states = [env.observation_space.sample() for i in range(n)]
            states = [obs_space_small.sample() for i in range(n)]
            naf.coldstart(states, targets)
            states = [obs_space_small.sample() for i in range(n)]
            actions = naf.actions(states)
            if np.all(np.less(actions, high)) and np.all(np.greater(actions, low)):
                success = True
                break
        if not success:
            print "WARNING: actions not in range"
        return naf


class NAFApproximation(object):

    def to_semi_definite(self, M):
        diag = tf.sqrt(tf.exp(tf.matrix_diag_part(M)))
        L = tf.matrix_set_diag(M * self.mask, diag)
        return tf.matmul(L, tf.transpose(L))

    def __init__(self, nnv, nnp, nnq, low, high, actiondim, learning_rate, discount, keep_prob=1):
        self.discount = tf.constant(discount, dtype=tf.float32)
        self.low = tf.constant(low, dtype=tf.float32)
        self.high = tf.constant(high, dtype=tf.float32)
        self.vx = nnv.x

        self.v = nnv.out
        self._setup_p_calculation(nnp, actiondim)
        self._setup_q_calculation(nnq, actiondim)
        self._setup_next_q_calulcation()
        self._setup_train_step(learning_rate)
        self.keep_prob = keep_prob

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _setup_p_calculation(self, nn, actiondim):
        mask = np.ones((actiondim, actiondim))
        mask[np.triu_indices(actiondim)] = 0
        self.mask = tf.constant(mask, dtype=tf.float32)
        self.px = nn.x

        upper_triang = tf.exp(
            tf.contrib.distributions.fill_triangular(nn.out))
        diag = tf.matrix_diag_part(upper_triang)
        L = tf.matrix_set_diag(upper_triang * mask, diag)
        self.P = tf.matmul(L, tf.transpose(L, perm=[0, 2, 1]))

    def _setup_q_calculation(self, nn, actiondim):
        self.action_inp = tf.placeholder(
            tf.float32, [None, actiondim], name="action")
        self.qx = nn.x
        self.mu = nn.out
        print "DIMENSION", self.mu

        self.batch = tf.reshape(self.action_inp - self.mu, [-1, 1, actiondim])
        self.a = tf.reshape(tf.matmul(
            tf.matmul(self.batch, self.P), tf.transpose(self.batch, [0, 2, 1])), [-1, 1])
        self.Q = self.v - .5 * self.a

        # coldstart action
        self.target_action = tf.placeholder(
            tf.float32, [None, actiondim], name="target_action")
        coldstart_loss = tf.reduce_sum(tf.square(self.target_action - self.mu))
        self.coldstart_actions = tf.train.AdamOptimizer(
            learning_rate=.1).minimize(coldstart_loss)

    def _setup_next_q_calulcation(self):
        self.r = tf.placeholder(tf.float32, [None, 1], name="reward")
        self.terminal = tf.placeholder(tf.float32, [None, 1])

        self.update = self.v * self.terminal * self.discount + self.r

    def _setup_train_step(self, learning_rate):
        self.target = tf.placeholder(tf.float32, [None, 1])
        self.actionloss = tf.reduce_sum(tf.abs(tf.to_float(tf.greater(self.mu, self.high)) * self.mu + tf.to_float(tf.less(
            self.mu, self.low)) * self.mu)) * 100
        self.loss = tf.reduce_sum(
            tf.square(self.target - self.Q))

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss + self.actionloss)

    def checkpoint(self, checkpoint_file):
        saver = tf.train.Saver()
        saver.save(self.session, checkpoint_file)

    def restore(self, checkpoint_file):
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_file)

    def calcA(self, x, action):
        return self.session.run(self.a, feed_dict={self.px: x, self.qx: x, self.action_inp: action, nn.keep_prob: 1.0})

    def calcP(self, x):
        return self.session.run(self.P, feed_dict={self.px: x, nn.keep_prob: 1.0})

    def value(self, x):
        return self.session.run(self.v, feed_dict={self.vx: x, nn.keep_prob: self.keep_prob})

    def actions(self, x):
        return self.session.run(self.mu, feed_dict={self.qx: x, nn.keep_prob: 1.0})

    def action(self, x, explore=False):
        action = self.session.run(
            self.mu, feed_dict={self.qx: x, nn.keep_prob: 1.0})[0]
        if explore:
            return action + np.random.normal()
        else:
            return action

    def calcq(self, rewards, next_state, terminals):
        return self.session.run(self.update, feed_dict={self.r: rewards, self.vx: next_state, self.terminal: terminals, nn.keep_prob: self.keep_prob})

    def storedq(self, state, action):
        return self.session.run(self.Q, feed_dict={self.vx: state, self.px: state, self.qx: state, self.action_inp: action, nn.keep_prob: 1.0})

    def calcloss(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.loss, feed_dict={self.target: target, self.vx: state, self.px: state, self.qx: state, self.action_inp: action, nn.keep_prob: self.keep_prob})

    def coldstart(self, state, target_action):
        return self.session.run(self.coldstart_actions, feed_dict={self.qx: state, self.target_action: target_action, nn.keep_prob: self.keep_prob})

    def trainstep(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.vx: state, self.px: state, self.qx: state, self.action_inp: action, nn.keep_prob: self.keep_prob})

    def __exit__(self):
        self.session.close()

    def renderBestA(self):
        states = np.expand_dims(np.arange(-20, 40, 10), axis=1)
        actions = self.actions(states)
        import matplotlib.pyplot as plt

        plt.plot(states, actions)
        plt.title("Best Actions")
        plt.waitforbuttonpress(0)
        plt.close()

    def renderA(self):
        actions = np.expand_dims(np.arange(-4, 4, .5), axis=1)
        states = np.expand_dims(np.arange(-10, 35, 15), axis=1)

        import matplotlib.pyplot as plt
        for state in states:
            value = self.calcA([state for a in actions], actions)
            plt.plot(actions, -5 * value, label=str(state))
        plt.legend()
        plt.waitforbuttonpress(0)
        plt.close()

    def renderQ(self, xx=None, rewards=None):
        x = np.expand_dims(np.arange(-20, 40, .5), axis=1)
        y = self.value(x)
        actions = [-2, 0, 2]

        import matplotlib.pyplot as plt
        for action in actions:
            q = self.storedq(x, [[action] for s in x])
            plt.plot(x, q, label=str(action))
        if xx:
            plt.plot(xx, rewards, 'ro')
        plt.plot(x, y, label='value')
        plt.legend()
        plt.waitforbuttonpress(0)
        plt.close()

    def renderV(self, x, rewards=None):
        # xx = [v for v in x]
        # xx.sort()
        # if len(xx) == 1:
        h = max(40, max(x)[0])
        l = min(-40, min(x)[0])
        xx = np.expand_dims(np.arange(l, h, .5), axis=1)
        y = self.value(xx)

        import matplotlib.pyplot as plt
        plt.plot(xx, y)
        if rewards:
            plt.plot(x, rewards, 'ro')
        plt.title("v over range")
        plt.waitforbuttonpress(0)
        plt.close()
