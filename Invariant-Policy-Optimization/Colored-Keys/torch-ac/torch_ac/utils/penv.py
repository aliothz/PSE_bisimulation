import gym


class ParallelEnv(gym.Env):
    """
    Keep the original batched-environment interface,
    but execute every environment sequentially.
    No multiprocessing is used.
    """

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        results = []

        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)

            if done:
                obs = env.reset()

            results.append((obs, reward, done, info))

        return zip(*results)

    def render(self):
        raise NotImplementedError