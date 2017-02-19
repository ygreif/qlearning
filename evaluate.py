import tempfile
import argparse

from gym import wrappers
import gym
import tensorflow as tf

from trainer import trainer
from agents import deep_q_agent


def evaluate(env, parameters, dir='/tmp/cartpole-experiment-1'):
    env = gym.wrappers.Monitor(env, dir, force=True)
    agent = deep_q_agent.DeepQAgent(env, parameters)
    for epoch in range(500):
        reward = agent.train_epoch(env, target_reward=500)
        if epoch % 100 == 0:
            print "Epcoh {} Reward {}".format(epoch, reward)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep q learning')
    parser.add_argument('--env', dest='env', default='CartPole-v0')
    parser.add_argument('--dir', dest='dir', default=tempfile.mkdtemp())
    parser.add_argument(
        '--hidden_layers', dest='hidden_layers', nargs='+', type=int, default=[190, 174, 72])
    parser.add_argument(
        '--memory_size', dest='memory_size', type=int, default=10000)
    parser.add_argument(
        '--minibatch_size', dest='minibatch_size', type=int, default=1000)
    parser.add_argument('--eps', dest='eps', type=float, default=.096)
    parser.add_argument(
        '--learning_rate', dest='learning_rate', type=float, default=.001)

    args = parser.parse_args()

    params = trainer.Parameters(
        learning_rate=args.learning_rate, hidden_layers=args.hidden_layers,
        memory_size=args.memory_size, minibatch_size=args.minibatch_size, eps=args.eps)
    print "Using parameters", params
    print "Saving to", args.dir
    env = gym.make(args.env)
    evaluate(env, params, args.dir)
