import gym


class Trainer(object):

    def __init__(self, env_name, agent):
        self.env = gym.make(env_name)
        self.agent = agent

    def train(self, target_reward, max_epochs=200):
        self.agent.register_env(self.env)

        for _ in range(20):
            reward = self.agent.train_epoch(
                self.env, target_reward, learn=False)
            print reward
        for _ in range(1000):
            self.agent.learn()
            self.agent.sample()
        print "Now for the main event"
        for epoch in range(max_epochs):
            reward = self.agent.train_epoch(self.env, target_reward)
            if epoch % 50 == 0:
                loss = self.agent.loss()
                print "Epoch {} Reward {} Loss {}".format(epoch, reward, loss)
            if reward >= target_reward:
                break

        return reward, epoch

    def run(self, max_steps=1000):
        self.agent.new_epoch()
        state = self.env.reset()
        step = 0
        while True:
            step += 1
            self.env.render()
            state, reward, done, _ = self.env.step(self.agent.action(state))
            if done or (max_steps and step >= max_steps):
                break
        print "Episode finished after {} timesteps".format(step)
