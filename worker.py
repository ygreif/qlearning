from concurrent.futures import ProcessPoolExecutor

import gym
from trainer import naftrainer
from agents.agent import NoExploration


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
        params = naftrainer.random_parameters()
        args.append((name, params, num_runs, max_epochs, i))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        r = [r for r in executor.map(helper, args)]
    return r


if __name__ == '__main__':
    print runworkers('Pendulum-v0', 1, 1, 1, 1)
