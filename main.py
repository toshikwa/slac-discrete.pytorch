import os
import yaml
import argparse
from datetime import datetime

from slac_discrete.env import make_pytorch_env
from slac_discrete.agent import SlacDiscreteAgent


def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(
        args.env_id, scale=False, gray_scale=False, image_size=64)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False,
        scale=False, gray_scale=False, image_size=64)

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'Slac-Discrete-{args.seed}-{time}')

    # Create the agent and run.
    agent = SlacDiscreteAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str,
        default=os.path.join('config', 'slac_discrete.yaml'))
    parser.add_argument(
        '--env_id', type=str, default='BattleZoneNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
