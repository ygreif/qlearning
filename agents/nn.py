import tensorflow as tf


class FullyConnectedLayer(object):

    def __init__(self, inp, dim):
        self.W = tf.Variable(tf.random_normal(dim))
        self.b = tf.Variable(tf.zeros(dim[1]))
        self.out = tf.matmul(inp, self.W) + self.b


class NeuralNetwork(object):

    def __init__(self, indim, enddim, hidden_layers):
        self.layers = []
        self.x = tf.placeholder(tf.float32, [None, indim])

        inp = self.x
        prev_dim = indim
        for out_dim in hidden_layers:
            self.layers.append(
                FullyConnectedLayer(inp, (prev_dim, out_dim)))
            inp = self.layers[-1].out
            prev_dim = out_dim
        self.layers.append(FullyConnectedLayer(inp, (prev_dim, enddim)))
        self.out = self.layers[-1].out


class NeuralAgent(object):

    def __init__(self, nn, discount=.95):
        self.discount = tf.constant(discount)

        self.x = nn.x
        self.act = tf.argmax(nn.out, axis=1)
        self.q = tf.transpose(tf.reduce_max(nn.out, axis=1))

        self.r = tf.placeholder(tf.float32, [None, 1])
        self.terminal = tf.placeholder(tf.float32, [None, 1])

        self.update = self.q * \
            tf.transpose(self.terminal * self.discount) + tf.transpose(self.r)

        self.target = tf.transpose(tf.placeholder(tf.float32, [None, 1]))
        self.loss = tf.reduce_sum(tf.square(self.target - self.q))
        self.train_step = tf.train.AdamOptimizer(
            learning_rate=0.01).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def action(self, state):
        return self.session.run(self.act, feed_dict={self.x: state})[0]

    def calcq(self, rewards, next_state, terminals):
        return self.session.run(self.update, feed_dict={self.r: rewards, self.x: next_state, self.terminal: terminals})

    def storedq(self, state):
        return self.session.run(self.q, feed_dict={self.x: state})

    def calcloss(self, state, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.loss, feed_dict={self.target: target, self.x: state})

    def trainstep(self, state, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.x: state})

    def __exit__(self):
        self.session.close()
