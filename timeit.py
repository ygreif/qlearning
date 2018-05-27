import tensorflow as tf
import gym
import gym.wrappers
from trainer import naftrainer

params = {'naf': {'learningParameters': {'discount': 0.99,
                                         'learning_rate': 0.001,
                                         'keep_prob': .9},
                  'nnpParameters': {'hidden_layers': [1, 1],
                                    'nonlinearity': tf.nn.tanh,
                                    'use_dropout': False},
                  'nnqParameters': {'hidden_layers': [400, 400],
                                    'nonlinearity': tf.nn.tanh,
                                    'use_dropout': True},
                  'nnvParameters': {'hidden_layers': [800, 800],
                                    'nonlinearity': tf.nn.tanh,
                                    'use_dropout': True}},
          'strat': {'decay': 800, 'scale': 1}}

params['strat']['max_steps'] = 200
params['strat']['target'] = 4000

env = gym.make('Pendulum-v0')
import time
s = time.time()
agent, reward, plots = naftrainer.train(
    env, params, max_epochs=1, writeevery=1, validate=False)
e = time.time()
print s, e, s - e
