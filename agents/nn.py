import numpy as np
import tensorflow as tf
import math


class FullyConnectedLayer(object):

    def __init__(self, inp, dim, nonlinearity=False, init='normal', init_bias=0.0):
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

    def __init__(self, indim, enddim, hidden_layers, nonlinearity=False, init='normal', init_bias=0.0):
        self.layers = []
        self.x = tf.placeholder(tf.float32, [None, indim], name="instate")
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

    def to_semi_definite(self, M):
#        diag = tf.sqrt(tf.exp(tf.matrix_diag_part(M)))
#        L = tf.matrix_set_diag(M * self.mask, diag)
#        return tf.matmul(L, tf.transpose(L))
        return self.mask

    def __init__(self, nnv, nnp, nnq, lower, upper, actiondim, learning_rate, discount):
        self.discount = tf.constant(discount)
        self.lower = tf.constant(lower)
        self.upper = tf.constant(upper)
        self.vx = nnv.x

        self.v = nnv.out
        self._setup_p_calculation(nnp, actiondim)
        self._setup_q_calculation(nnq, actiondim)
        self._setup_next_q_calulcation()
        self._setup_train_step(learning_rate)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _setup_p_calculation(self, nn, actiondim):
#        mask = np.ones((actiondim, actiondim))
        mask = np.zeros((actiondim, actiondim))
        np.fill_diagonal(mask, 1.0)
#        mask[np.triu_indices(actiondim)] = 0
        self.mask = tf.constant(mask, dtype=tf.float32)
        self.px = nn.x
#        self.layer = FullyConnectedLayer(
#            nn.out, (nn.enddim, actiondim * actiondim))
        self.P = tf.map_fn(self.to_semi_definite,
                           tf.reshape(nn.out, (-1, actiondim, actiondim)))

    def _setup_q_calculation(self, nn, actiondim):
        self.action_inp = tf.placeholder(
            tf.float32, [None, actiondim], name="action")
        self.qx = nn.x
        self.mu = nn.out

        self.batch = tf.reshape(self.action_inp - self.mu, [-1, 1, actiondim])
        self.a = tf.reshape(tf.batch_matmul(
            tf.batch_matmul(self.batch, self.P), tf.transpose(self.batch, [0, 2, 1])), [-1, 1])
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
        self.actionloss = tf.reduce_sum(tf.abs(tf.to_float(tf.greater(self.mu, self.upper)) * self.mu + tf.to_float(tf.less(
            self.mu, self.lower)) * self.mu))
        self.loss = tf.reduce_sum(
            tf.square(self.target - self.Q))

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss + self.actionloss)

    def calcA(self, x, action):
        return self.session.run(self.a, feed_dict={self.px: x, self.qx: x, self.action_inp: action})

    def calcP(self, x):
        return self.session.run(self.P, feed_dict={self.px: x})

    def value(self, x):
        return self.session.run(self.v, feed_dict={self.vx: x})

    def actions(self, x):
        return self.session.run(self.mu, feed_dict={self.qx: x})

    def action(self, x, explore=False):
        action = self.session.run(self.mu, feed_dict={self.qx: x})[0]
        if explore:
            return action + np.random.normal()
        else:
            return action

    def calcq(self, rewards, next_state, terminals):
        return self.session.run(self.update, feed_dict={self.r: rewards, self.vx: next_state, self.terminal: terminals})

    def storedq(self, state, action):
        return self.session.run(self.Q, feed_dict={self.vx: state, self.px: state, self.qx: state, self.action_inp: action})

    def calcloss(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        print "target", target
        return self.session.run(self.loss, feed_dict={self.target: target, self.vx: state, self.px: state, self.qx: state, self.action_inp: action})

    def coldstart(self, state, target_action):
        return self.session.run(self.coldstart_actions, feed_dict={self.qx: state, self.target_action: target_action})

    def trainstep(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.vx: state, self.px: state, self.qx: state, self.action_inp: action})

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

'''
if __name__ == '__main__':
    nn = NeuralNetwork(2, 2, [1], nonlinearity=False,
                       init='normal', init_bias=0)
    naf = NAFApproximation(nn, 3, .001, .99)
#    control = naf.action([[0, 0]])
    print "Pshape", naf.calcP([[0, 0], [0, 0]])
    print naf.value([[0, 0]])
    print "calcq", naf.calcq([[10], [0], [10]], [[1, 1], [1, 1], [1, 1]], [[0], [1], [1]])
    print "storedq", naf.storedq([[1, 2], [3, 3]], [[0, 0, 0], [1, 1, 1]])
'''
# print "calcqshape", naf.calcq([[10, 10]], [[1, 1], [2, 2]], [[0,
# 0]]).shape
#    print "control", control


class QApproximation(object):

    def __init__(self, nn, parameters, discount=.95):
        self.discount = tf.constant(discount)
        self.choices = range(nn.enddim)

        self._setup_q_calculation(nn)
        self._setup_next_q_calulcation(nn)
        self._setup_train_step(parameters)

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

    def _setup_train_step(self, parameters):
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

    def calcloss(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.loss, feed_dict={self.target: target, self.x: state})

    def trainstep(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.x: state})

    def __exit__(self):
        self.session.close()
