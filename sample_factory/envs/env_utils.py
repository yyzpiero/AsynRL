from sample_factory.utils.utils import is_module_available, log
from functools import wraps
from time import sleep
#from sample_factory.algo.utils.context import global_env_registry

class EnvCriticalError(Exception):
    pass


# def register_env(env_name: str, make_env_func: CreateEnvFunc) -> None:
#     """
#     Register a callable that creates an environment.
#     This callable is called like:
#         make_env_func(full_env_name, cfg, env_config)
#         Where full_env_name is the name of the environment to be created, cfg is a namespace or AttrDict containing
#         necessary configuration parameters and env_config is an auxiliary dictionary containing information such as worker index on which the environment lives
#         (some envs may require this information)
#     env_name: name of the environment
#     make_env_func: callable that creates an environment
#     """

#     env_registry = global_env_registry()

#     if env_name in env_registry:
#         log.warning(f"Environment {env_name} already registered, overwriting...")

#     assert callable(make_env_func), f"{make_env_func=} must be callable"

#     env_registry[env_name] = make_env_func


def vizdoom_available():
    return is_module_available('vizdoom')


def minigrid_available():
    return is_module_available('gym_minigrid')


def dmlab_available():
    return is_module_available('deepmind_lab')


def retry(exception_class=Exception, num_attempts=3, sleep_time=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(num_attempts):
                try:
                    return func(*args, **kwargs)
                except exception_class as e:
                    if i == num_attempts - 1:
                        raise
                    else:
                        log.error('Failed with error %r, trying again', e)
                        sleep(sleep_time)

        return wrapper

    return decorator


def find_wrapper_interface(env, interface_type):
    """Unwrap the env until we find the wrapper that implements interface_type."""
    unwrapped = env.unwrapped
    while True:
        if isinstance(env, interface_type):
            return env
        elif env == unwrapped:
            return None  # unwrapped all the way and didn't find the interface
        else:
            env = env.env  # unwrap by one layer


class RewardShapingInterface:
    def __init__(self):
        pass

    def get_default_reward_shaping(self):
        """Should return a dictionary of string:float key-value pairs defining the current reward shaping scheme."""
        raise NotImplementedError

    def get_current_reward_shaping(self, agent_idx: int):
        raise NotImplementedError

    def set_reward_shaping(self, reward_shaping: dict, agent_idx: int):
        """
        Sets the new reward shaping scheme.
        :param reward_shaping dictionary of string-float key-value pairs
        :param agent_idx: integer agent index (for multi-agent envs)
        """
        raise NotImplementedError


def get_default_reward_shaping(env):
    """
    The current convention is that when the environment supports reward shaping, the env.unwrapped should contain
    a reference to the object implementing RewardShapingInterface.
    We use this object to get/set reward shaping schemes generated by PBT.
    """

    reward_shaping_interface = find_wrapper_interface(env, RewardShapingInterface)
    if reward_shaping_interface:
        return reward_shaping_interface.get_default_reward_shaping()

    return None


def set_reward_shaping(env, reward_shaping: dict, agent_idx: int):
    reward_shaping_interface = find_wrapper_interface(env, RewardShapingInterface)
    if reward_shaping_interface:
        reward_shaping_interface.set_reward_shaping(reward_shaping, agent_idx)


class TrainingInfoInterface:
    def __init__(self):
        self.training_info = dict()

    def set_training_info(self, training_info):
        """
        Send the training information to the environment, i.e. number of training steps so far.
        Some environments rely on that i.e. to implement curricula.
        :param training_info: dictionary containing information about the current training session. Guaranteed to
        contain 'approx_total_training_steps' (approx because it lags a bit behind due to multiprocess synchronization)
        """
        self.training_info = training_info


def find_training_info_interface(env):
    """Unwrap the env until we find the wrapper that implements TrainingInfoInterface."""
    return find_wrapper_interface(env, TrainingInfoInterface)


def set_training_info(training_info_interface, approx_total_training_steps: int):
    if training_info_interface:
        training_info_dict = dict(approx_total_training_steps=approx_total_training_steps)
        training_info_interface.set_training_info(training_info_dict)



