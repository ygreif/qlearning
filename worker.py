from concurrent.futures import ProcessPoolExecutor

import gym
from trainer import naftrainer
from agents.agent import NoExploration

import tensorflow as tf
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


def worker(name, params, attempts, max_epochs, num):
    env = gym.make(name)
    rewards = []
    for attempt in range(attempts):
        print "Param set", num, "attempt", attempt
        agent, _, _ = naftrainer.train(env, params, max_epochs=max_epochs)

        eval_strat = NoExploration(max_steps=500)
        reward = 0
        for _ in range(10):
            reward += agent.train_epoch(env, eval_strat, learn=False)
        rewards.append(reward / 10.0)
    print "Returning"
    return [params, rewards]


def helper(args):
    return worker(*args)


def runworkers(name, num_params, max_epochs, num_runs, max_workers):
    args = []
    for i in range(num_params):
#        params = naftrainer.random_parameters()
        args.append((name, params, num_runs, max_epochs, i))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        r = [r for r in executor.map(helper, args)]
    return r


if __name__ == '__main__':
    import timeit
    print timeit.timeit("runworkers('Pendulum-v0', 1, 1, 1, 1)", setup="from __main__ import runworkers")
