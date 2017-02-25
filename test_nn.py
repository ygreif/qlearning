import numpy
import numpy as np
import tensorflow as tf
import agents.nn
import agents.memory
import math

mem = agents.memory.Memory(100000)
nnv = agents.nn.NeuralNetwork(1, 1, [10], nonlinearity=tf.nn.relu)
nnp = agents.nn.NeuralNetwork(1, 1, [1], nonlinearity=tf.nn.tanh)
nnq = agents.nn.NeuralNetwork(1, 1, [10], nonlinearity=tf.nn.relu)
print "Setting up NAF"
naf = agents.nn.NAFApproximation(nnv, nnp, nnq, 1, .1, 0.5)
state = 0
print "Start"
print "action0", naf.action([[0]], explore=False)
print "action20", naf.action([[20]], explore=False)
print "action30", naf.action([[30]], explore=False)
print "action10", naf.action([[10]], explore=False)

naf.renderBestA()
for i in range(100):
    x = np.expand_dims(np.random.uniform(-40, 40, 100), axis=1)
    target = [[0] for _ in x]
    naf.coldstart(x, target)
naf.renderBestA()

for i in range(10000):
    action = naf.action([[state]], explore=True)[0]
    prev_state = state
    state = state + action + numpy.random.normal(0) / 4.0
    if state < -40:
        state = -40
    elif state > 60:
        state = 60
    reward = 10 - 1 * abs(state - 20)
    if action > 2:
        reward -= 10.0 * action
    elif action < -2:
        reward += 10.0 * action
    else:
        reward += (10 - abs(action))
    mem.append([prev_state], [action], reward, [state], False)
    if i % 100 == 0:
        value = naf.value([[prev_state]])
        Q = naf.storedq([[prev_state]], [[action]])
        P = naf.calcP([[prev_state]])
        print "Iter", i, "state", state, "action", action, "reward", reward, "value", value, "Q", Q, "prev_state", prev_state, "P", P
        trainstate, actions, rewards, next_state, terminals = mem.minibatch(
            100)
#        x = np.expand_dims(, axis=1)
        naf.renderV(trainstate, rewards)
#        naf.renderA()
        naf.renderQ(trainstate, rewards)
        naf.renderBestA()
#        print mem.minibatch(1)
    trainstate, actions, rewards, next_state, terminals = mem.minibatch(100)
    if i > 100:
        naf.trainstep(trainstate, actions, rewards, next_state, terminals)
print naf.action([[state]], explore=False)
print naf.value([[state]])

print "End"
print "action-1", naf.action([[state]], explore=False), state
print "action0", naf.action([[0]], explore=False)
print "action20", naf.action([[20]], explore=False)
print "action30", naf.action([[30]], explore=False)
print "action10", naf.action([[10]], explore=False)
for i in range(-20, 60):
    print "state", i, "action", naf.action([[i]])
