import numpy as np
import tensorflow as tf

import agents.naf
import agents.agent
import agents.validation


def random_nn_parameters(hidden_choices=[[5], [10], [100], [200], [500], [500, 500], [500, 500, 500], [200, 200], [1000], [50], [10, 10], [10, 10, 10], [100, 100, 100], [100, 100]], nonlinearity_choices=[tf.nn.relu, tf.nn.tanh]):
    return {'hidden_layers': np.random.choice(hidden_choices), 'nonlinearity': np.random.choice(nonlinearity_choices)}


def random_learning_parameters(rates=[.001, .002, .0001], discount=[.75, .95]):
    return {'learning_rate': np.random.choice(rates), 'discount': np.random.choice(discount)}


def random_strat(scale=[.1, .5, 1], decay=[50, 100, 200, 1000, 100000]):
    return {'scale': np.random.choice(scale), 'decay': np.random.choice(decay)}


def random_parameters():
    nnvParameters = random_nn_parameters()
    nnpchoices = [[1], [5], [10], [10, 10], [1, 1]]
    nnpParameters = random_nn_parameters(
        hidden_choices=nnpchoices)
    nnqParameters = random_nn_parameters()
    learningParameters = random_learning_parameters()
    strat = random_strat()
    parameters = {'naf': {
        'nnvParameters': nnvParameters, 'nnpParameters': nnpParameters,
        'nnqParameters': nnqParameters, 'learningParameters': learningParameters}, 'strat': strat}
    return parameters


def train(env, parameters, max_epochs=400, writeevery=100, validate=False, validation_env=False):
    if not validation_env:
        validation_env = env

    naf = agents.naf.SetupNAF.setup(env, **parameters['naf'])
    a = agents.agent.Agent(naf)
    plots = []
    for epoch in range(max_epochs):
        strat = agents.agent.DeltaStrategy(epoch=epoch, **parameters['strat'])
        r = a.train_epoch(env, strat)
        if epoch % writeevery == 0:
            print "Epoch", epoch, "Reward", r
            if validate and epoch > 0:
                insample, outsample, onstrat = agents.validation.validate(
                    validation_env, a, 5000, 60, log=False)
                agents.validation.plot(insample, outsample, onstrat)
    return a, r, plots
