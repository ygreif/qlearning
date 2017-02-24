import numpy
import tensorflow as tf
import agents.nn
import agents.memory
import math

mem = agents.memory.Memory(1000)
nnv = agents.nn.NeuralNetwork(1, 1, [20, 20], nonlinearity=tf.nn.tanh)
nnp = agents.nn.NeuralNetwork(1, 1, [5, 5], nonlinearity=tf.nn.tanh)
nna = agents.nn.NeuralNetwork(1, 1, [20, 20], nonlinearity=tf.nn.tanh)
print "Setting up NAF"
naf = agents.nn.NAFApproximation(nnv, 1, .001, .7)
state = 0
print "Start"
print "action0", naf.action([[0]], explore=False)
print "action20", naf.action([[20]], explore=False)
print "action30", naf.action([[30]], explore=False)
print "action10", naf.action([[10]], explore=False)

for i in range(10000):
    action = naf.action([[state]], explore=True)[0]
    prev_state = state
    state = state + action + numpy.random.normal(0) / 4.0
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
