import gym
import gym_maze


def make_nasim_env(env_name, cfg=None, **kwargs):
    env = gym.make(env_name)
    return env

