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
    onstrat_memory = agent.memory

    agent.memory = oldmemory

    checkpoint_file = tempfile.NamedTemporaryFile(delete=False)
    checkpoint_name = checkpoint_file.name
    checkpoint_file.close()

    q = agent.q
    q.checkpoint(checkpoint_name)
    for i in range(iters):
        if log:
            print "On iteration", i
        for mem, losses in [(insample_memory, insample_loss), (outsample_memory, outsample_loss), (onstrat_memory, onstrat_loss)]:
            minibatch = mem.minibatch(samples)
            losses.append(q.calcloss(minibatch.state, minibatch.actions,
                                     minibatch.rewards, minibatch.next_state, minibatch.terminals))
        insample_minibatch = insample_memory.minibatch(samples)
        q.trainstep(insample_minibatch.state, insample_minibatch.actions,
                    insample_minibatch.rewards, insample_minibatch.next_state, insample_minibatch.terminals)
    q.restore(checkpoint_name)
    return insample_loss, outsample_loss, onstrat_loss
