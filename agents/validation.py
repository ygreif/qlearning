import tempfile

from agent import NoExploration
import memory


def validate(env, agent, samples, iters, log=False):
    insample_loss = []
    outsample_loss = []
    onstrat_loss = []

    oldmemory = agent.memory
    agent.memory = memory.Memory(samples)
    while len(agent.memory) < samples:
        if log:
            print "onstrat memory", len(agent.memory)
        strat = NoExploration()
        agent.train_epoch(env, strat, learn=False)

    insample_memory, outsample_memory = oldmemory.split()
    insample_minibatch = insample_memory.minibatch(samples)
    outsample_minibatch = outsample_memory.minibatch(samples)
    onstrat_minibatch = agent.memory.minibatch(samples)

    agent.memory = oldmemory

    checkpoint_file = tempfile.NamedTemporaryFile(delete=False)
    checkpoint_name = checkpoint_file.name
    checkpoint_file.close()

    q = agent.q
    q.checkpoint(checkpoint_name)
    for i in range(iters):
        if log:
            print "On iteration", i
        for minibatch, losses in [(insample_minibatch, insample_loss), (outsample_minibatch, outsample_loss), (onstrat_minibatch, onstrat_loss)]:
            losses.append(q.calcloss(minibatch.state, minibatch.actions,
                                     minibatch.rewards, minibatch.next_state, minibatch.terminals))
        # handle insample
        minibatch = insample_memory.minibatch(samples)
        q.trainstep(minibatch.state, minibatch.actions,
                    minibatch.rewards, minibatch.next_state, minibatch.terminals)
    q.restore(checkpoint_name)
    return insample_loss, outsample_loss, onstrat_loss


def plot(insample, outsample, onstrat):
    import matplotlib.pyplot as plt
    plt.plot(insample, label="in sample")
    plt.plot(outsample, label="out sample")
    plt.plot(onstrat, label="on strat")
    plt.legend()
    plt.show()

    def normalize(x):
        h = x[0]
        return [elem / h for elem in x]

    plt.plot(normalize(insample), label="in sample")
    plt.plot(normalize(outsample), label="out sample")
    plt.plot(normalize(onstrat), label="on strat")
    plt.legend()
    plt.show()
