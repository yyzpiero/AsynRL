import gym


def make_nasim_env(env_name, cfg, **kwargs):
    assert env_name.startswith('nasim_')
    nasim_env_name = env_name.split('nasim_')[1]

    env = gym.make("nasim:"+nasim_env_name)
    
    return env