import argparse

from .cli.train import setup as train_setup
from .cli.eval import setup as eval_setup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_setup(subparsers.add_parser("train"))
    eval_setup(subparsers.add_parser("eval"))
    args = parser.parse_args()
    args.func(args)
