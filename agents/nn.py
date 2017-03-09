import tensorflow as tf
import math

epsilon = 1e-3


class NewFullyConnectedLayer(object):

    def __init__(self, inp, dim, nonlinearity=False, keep_prob=1.0, batch_norm=False):
        self.W = tf.Variable(tf.random_normal(dim))
        self.b = tf.Variable(tf.constant(1.0, shape=(1, dim[1])))
        self.train_keep_prob = tf.constant(keep_prob, tf.float32)
        self.test_keep_prob = tf.constant(1.0, tf.float32)

        # training
        if batch_norm:
            decay = tf.constant(batch_norm, tf.float32)
            scale = tf.Variable(tf.ones([inp.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([inp.get_shape()[-1]]))

            pop_mean = tf.Variable(
                tf.zeros([inp.get_shape()[-1]]), trainable=False)
            pop_var = tf.Variable(
                tf.ones([inp.get_shape()[-1]]), trainable=False)

            batch_mean, batch_var = tf.nn.moments(inp, [0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                inptrain = tf.nn.batch_normalization(
                    inp, batch_mean, batch_var, beta, scale, epsilon)
        else:
            inptrain = inp
        if nonlinearity:
            htrain = nonlinearity(tf.matmul(inptrain, self.W) + self.b)
        else:
            htrain = tf.matmul(inptrain, self.W) + self.b
        if keep_prob < 1:
            self.trainout = tf.nn.dropout(htrain, self.train_keep_prob)
        else:
            self.trainout = htrain

        # test
        if batch_norm:
            inptrain = tf.nn.batch_normalization(
                inp, pop_mean, pop_var, beta, scale, epsilon)
        else:
            inptrain = inp
        if nonlinearity:
            htest = nonlinearity(tf.matmul(inp, self.W) + self.b)
        else:
            htest = tf.matmul(inp, self.W) + self.b
        if keep_prob < 1:
            self.testnout = tf.nn.dropout(htest, self.test_keep_prob)
        else:
            self.testout = htrain


keep_prob = tf.placeholder(tf.float32)


class FullyConnectedLayer(object):

    def __init__(self, inp, dim, nonlinearity=False, init='normal', init_bias=0.0, use_dropout=False):
        if init == 'normal':
            self.W = tf.Variable(tf.random_normal(dim))
            self.b = tf.Variable(tf.constant(1.0, shape=(1, dim[1])))
        elif init == 'uniform':
            bound = math.sqrt(6) / math.sqrt(dim[0] + dim[1])
            self.W = tf.Variable(
                tf.random_uniform(dim, minval=-1 * bound, maxval=bound))
            self.b = tf.Variable(tf.zeros(dim[1]))
        if nonlinearity:
            h = nonlinearity(tf.matmul(inp, self.W) + self.b)
        else:
            h = tf.matmul(inp, self.W) + self.b
        if use_dropout:
            self.out = tf.nn.dropout(h, keep_prob)
        else:
            self.out = h


class NeuralNetwork(object):

    def __init__(self, indim, enddim, hidden_layers=[10], nonlinearity=False, init='normal', init_bias=0.0, name='default', use_dropout=False):
        self.layers = []
        self.x = tf.placeholder(tf.float32, [None, indim], name=name)
        self.indim = indim
        self.enddim = enddim

        inp = self.x
        prev_dim = indim
        for out_dim in hidden_layers:
            self.layers.append(
                FullyConnectedLayer(inp, (prev_dim, out_dim), nonlinearity=nonlinearity, init=init, init_bias=0.0, use_dropout=use_dropout))
            inp = self.layers[-1].out
            prev_dim = out_dim
        self.layers.append(FullyConnectedLayer(
            inp, (prev_dim, enddim), nonlinearity=False, init_bias=init_bias))
        self.out = self.layers[-1].out


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
