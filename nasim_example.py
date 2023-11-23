import argparse
import os
from distutils.util import strtobool
from sample_factory.algorithms.appo.appo import APPO,A3C,IMPALA
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm

import sys
import torch
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"


"""
An example that shows how to use SampleFactory with a Gym env.

Example command line for CartPole-v1:
python -m sample_factory_examples.train_gym_env --algo=APPO --use_rnn=False --num_envs_per_worker=20 --policy_workers_per_policy=2 --recurrence=1 --with_vtrace=False --batch_size=512 --hidden_size=256 --encoder_type=mlp --encoder_subtype=mlp_mujoco --reward_scale=0.1 --save_every_sec=10 --experiment_summaries_interval=10 --experiment=example_gym_cartpole-v1 --env=gym_CartPole-v1
python -m sample_factory_examples.enjoy_gym_env --algo=APPO --experiment=example_gym_cartpole-v1 --env=gym_CartPole-v1

"""


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",  default="Pocp1Gen 35 Host Network", type=str,
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1223,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with tensorboard")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="nasim_Pocp1Gen-v0",
        help="the id of the environment")
    parser.add_argument("--num_workers", type=int, default=32,
        help="the id of the environment")
    parser.add_argument(
            '--num_envs_per_worker', default=8, type=int,
            help='Number of envs on a single CPU actor, in high-throughput configurations this should be in 10-30 range for Atari/VizDoom '
                    'Must be even for double-buffered sampling!')
    parser.add_argument("--no-obs-norm",type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="normali")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument('--encoder_type', default='mlp', type=str, help='Type of the encoder. Supported: conv, mlp, resnet (feel free to define more)')
    parser.add_argument('--encoder_subtype', default='mlp_nasim', type=str, help='Specific encoder design (see model.py)')
    args = parser.parse_args()
    return args

def main():
    since = time.time()
    args = parse_args()
    device = "gpu"
    #args.env_id = "maze-random-10x10-plus-v0"
    
    model = APPO(env=args.env_id, device=device, 
                num_workers=args.num_workers,
                num_envs_per_worker=args.num_envs_per_worker,
                encoder=args.encoder_type, 
                encodersubtype=args.encoder_subtype,
                experiment = args.experiment,

                policy_kwargs = {"num_policies":2,
                                "reward_scale":0.01,
                                "kl_loss_coeff":1.0,
                                "use_rnn":False,
                                "actor_critic_share_weights":False})
    
    model.train(train_for_env_steps=10000000)

    time_elapsed = time.time()-since
    print("Total Run Time: {}".format(time_elapsed))

if __name__ == '__main__':
    sys.exit(main())

    
