import collections
import numpy as np


class Memory(object):

    def __init__(self, size):
        self.memory = collections.deque(maxlen=size)

    def append(self, state, action, reward, next_state, done):
        self.memory.append(
            {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done})

    def minibatch(self, size):
        replays = np.random.choice(self.memory, size)

        state = [replay['state'] for replay in replays]
        action = [replay['action'] for replay in replays]
        rewards = [[replay['reward']] for replay in replays]
        next_state = [replay['next_state'] for replay in replays]
        terminals = [[0] if replay['done'] else [1] for replay in replays]
        return state, action, rewards, next_state, terminals
