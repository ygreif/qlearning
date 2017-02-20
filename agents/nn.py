import numpy as np
import tensorflow as tf
import math


class FullyConnectedLayer(object):

    def __init__(self, inp, dim, nonlinearity=False, init='normal', init_bias=0):
        if init == 'normal':
            self.W = tf.Variable(tf.random_normal(dim))
            self.b = tf.Variable(tf.constant(1.0, shape=(1, dim[1])))
        elif init == 'uniform':
            bound = math.sqrt(6) / math.sqrt(dim[0] + dim[1])
            self.W = tf.Variable(
                tf.random_uniform(dim, minval=-1 * bound, maxval=bound))
            self.b = tf.Variable(tf.zeros(dim[1]))
        if nonlinearity:
            self.out = nonlinearity(tf.matmul(inp, self.W) + self.b)
        else:
            self.out = tf.matmul(inp, self.W) + self.b


class NeuralNetwork(object):

    def __init__(self, indim, enddim, hidden_layers, nonlinearity, init, init_bias):
        self.layers = []
        self.x = tf.placeholder(tf.float32, [None, indim])
        self.indim = indim
        self.enddim = enddim

        inp = self.x
        prev_dim = indim
        for out_dim in hidden_layers:
            self.layers.append(
                FullyConnectedLayer(inp, (prev_dim, out_dim), nonlinearity=nonlinearity, init=init, init_bias=0.0))
            inp = self.layers[-1].out
            prev_dim = out_dim
        self.layers.append(FullyConnectedLayer(
            inp, (prev_dim, enddim), nonlinearity=False, init_bias=init_bias))
        self.out = self.layers[-1].out


class NAFApproximation(object):

    def __init__(self, nn, actiondim):
        self.x = nn.x
        self.v = FullyConnectedLayer(nn.out, (nn.enddim, 1)).out
        self._setup_p_calculation(nn, actiondim)
        self._setup_q_calculation(nn, actiondim)
        self._setup_next_q_calulcation(nn)
        self._setup_train_step(nn)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _setup_p_calculation(self, nn, actiondim):
        mask = np.ones((actiondim, actiondim))
        mask[np.triu_indices(actiondim), -1] = 0
        self.mask = tf.constant(mask)

        fullL = tf.reshape(FullyConnectedLayer(
            nn.out, (nn.enddim, actiondim * actiondim)).out, (actiondim, actiondim))
        diag = tf.exp(tf.matrix_diag_part(fullL))
        L = tf.matrix_set_diag(fullL * mask, diag)
        self.P = tf.matmul(L, tf.transpose(L))

    def _setup_q_calculation(self, nn, actiondim):
        self.action_inp = tf.placeholder(tf.float32, [None, actiondim])
        self.mu = FullyConnectedLayer(nn.out, (nn.enddim, actiondim)).out
        self.a = -.5 * tf.matmul(tf.matmul((self.action_inp - self.mu),
                                           self.P), tf.transpose(self.action_inp - self.mu))
        self.Q = self.v + self.a

    def _setup_next_q_calulcation(self, nn):
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.terminal = tf.placeholder(tf.float32, [None, 1])

        self.update = self.v * \
            tf.transpose(self.terminal * self.discount) + tf.transpose(self.r)

    def _setup_train_step(self, nn, parameters):
        self.target = tf.transpose(tf.placeholder(tf.float32, [None, 1]))
        self.loss = tf.reduce_sum(tf.square(self.target - self.q))

        if parameters.learner == 'adam':
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=parameters.learning_rate).minimize(self.loss)
        else:
            self.train_step = tf.train.GradientDescentOptimizer(
                learning_rate=parameters.learning_rate).minimize(self.loss)

    def value(self, x):
        return self.session.run(self.v, feed_dict={self.x: x})

    def action(self, x):
        return self.session.run(self.mu, feed_dict={self.x: x})

    def calcq(self, rewards, next_state, terminals):
        return self.session.run(self.update, feed_dict={self.r: rewards, self.x: next_state, self.terminal: terminals})

    def storedq(self, state):
        return self.session.run(self.q, feed_dict={self.x: state})

    def calcloss(self, state, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.loss, feed_dict={self.target: target, self.x: state})

    def trainstep(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.x: state, self.action_inp: action})

    def __exit__(self):
        self.session.close()


if __name__ == '__main__':
    nn = NeuralNetwork(2, 2, [100], nonlinearity=False,
                       init='normal', init_bias=0)
    naf = NAFApproximation(nn, 3)
    control = naf.control([[0, 0]])


class QApproximation(object):

    def __init__(self, nn, parameters, discount=.95):
        self.discount = tf.constant(discount)
        self.choices = range(nn.enddim)

        self._setup_q_calculation(nn)
        self._setup_next_q_calulcation(nn)
        self._setup_train_step(nn, parameters)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _setup_q_calculation(self, nn):
        self.x = nn.x
        self.qprobs = tf.nn.softmax(nn.out)
        self.act = tf.argmax(nn.out, axis=1)
        self.q = tf.transpose(tf.reduce_max(nn.out, axis=1))

    def _setup_next_q_calulcation(self, nn):
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.terminal = tf.placeholder(tf.float32, [None, 1])

        self.update = self.q * \
            tf.transpose(self.terminal * self.discount) + tf.transpose(self.r)

    def _setup_train_step(self, nn, parameters):
        self.target = tf.transpose(tf.placeholder(tf.float32, [None, 1]))
        self.loss = tf.reduce_sum(tf.square(self.target - self.q))

        if parameters.learner == 'adam':
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=parameters.learning_rate).minimize(self.loss)
        else:
            self.train_step = tf.train.GradientDescentOptimizer(
                learning_rate=parameters.learning_rate).minimize(self.loss)

    def action(self, state):
        return self.session.run(self.act, feed_dict={self.x: state})[0]

    def calcq(self, rewards, next_state, terminals):
        return self.session.run(self.update, feed_dict={self.r: rewards, self.x: next_state, self.terminal: terminals})

    def storedq(self, state):
        return self.session.run(self.q, feed_dict={self.x: state})

    def calcloss(self, state, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.loss, feed_dict={self.target: target, self.x: state})

    def trainstep(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.x: state})

    def __exit__(self):
        self.session.close()
