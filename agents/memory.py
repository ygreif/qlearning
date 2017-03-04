import collections
import numpy as np


Minibatch = collections.namedtuple(
    'Minibatch', ['state', 'actions', 'rewards', 'next_state', 'terminals'])


class Memory(object):

    def __init__(self, size):
        self.memory = collections.deque(maxlen=size)

    def append(self, state, action, reward, next_state, done):
        self.memory.append(
            {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done})

    def split(self):
        length = len(self.memory)
        idx_batch = list(
            np.random.choice(np.arange(0, len(self.memory)), length))
        idx_batch1 = idx_batch[0:length / 2]
        idx_batch2 = idx_batch[length / 2:length]
        mem1 = Memory(length)
        mem2 = Memory(length)
        for idx, mem in ((idx_batch1, mem1), (idx_batch2, mem2)):
            for replay in [val for i, val in enumerate(self.memory) if i in idx]:
                mem.append(**replay)
        return mem1, mem2

    def minibatch(self, size):
        idx_batch = set(
            np.random.choice(np.arange(0, len(self.memory)), size))
        replays = [val for i, val in enumerate(self.memory) if i in idx_batch]

        state = [replay['state'] for replay in replays]
        action = [replay['action'] for replay in replays]
        rewards = [[replay['reward']] for replay in replays]
        next_state = [replay['next_state'] for replay in replays]
        terminals = [[0] if replay['done'] else [1] for replay in replays]
        return Minibatch(state, action, rewards, next_state, terminals)

    def __len__(self):
        return len(self.memory)
