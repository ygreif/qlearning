import tensorflow as tf
import numpy as np

import neuralnetwork


class Actor(object):

    def __init__(self, nn, compress=False):
        self.max_prod = tf.placeholder(tf.float32, [None, 1], name="max_prod")

        self.x = nn.x
        if compress:
            self.mu = ((tf.tanh(nn.out) + 1.0) / 2.0) * self.max_prod
        else:
            self.mu = nn.out

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def action(self, x, max_prod):
        return self.session.run(
            self.mu, feed_dict={
                self.x: x, self.max_prod: max_prod,
                neuralnetwork.keep_prob: 1.0})[0]

    def actions(self, x, max_prod):
        return self.session.run(self.mu, feed_dict={self.x: x, self.max_prod: max_prod, neuralnetwork.keep_prob: 1.0})


class QApproximation(object):

    def __init__(self, nn, parameters, discount=.95):
        self.discount = tf.constant(discount)

        self._setup_q_calculation(nn)
        self._setup_next_q_calulcation(nn)
        self._setup_train_step(parameters)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _setup_q_calculation(self, nn):
        self.x = nn.x
        self.q = nn.out

    def _setup_next_q_calulcation(self, nn):
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.terminal = tf.placeholder(tf.float32, [None, 1])

        self.update = self.q * \
            tf.transpose(self.terminal * self.discount) + tf.transpose(self.r)

    def _setup_train_step(self, parameters):
        self.target = tf.transpose(tf.placeholder(tf.float32, [None, 1]))
        self.loss = tf.reduce_sum(tf.square(self.target - self.q))

        if parameters.learner == 'adam':
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=parameters.learning_rate).minimize(self.loss)
        else:
            self.train_step = tf.train.GradientDescentOptimizer(
                learning_rate=parameters.learning_rate).minimize(self.loss)

    def calcq(self, rewards, next_state, terminals):
        return self.session.run(self.update, feed_dict={self.r: rewards, self.x: next_state, self.terminal: terminals})

    def trainstep(self, state, action, rewards, next_state, next_action, terminals):
        actionstate = np.concatenate(state, action, axis=1)
        nextactionstate = np.concatenate(next_state, next_action, axis=1)
        target = self.calcq(rewards, nextactionstate, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.x: nextstate})

    def __exit__(self):
        self.session.close()


class ActorCritic(object):

    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic

    def action(self, x, max_prod):
        return self.actor.action(x, max_prod)

    def actions(self, x, max_prod):
        return self.actor.actions(x, max_prod)

    def trainstep(self, state, action, rewards, next_state, next_action, terminals):
        # TODO: add max_prod
        nextactions = self.actions(next_state, [n for n in next_state])
        self.critic.trainstep(
            state, action, rewards, next_state, nextactions, terminals)
