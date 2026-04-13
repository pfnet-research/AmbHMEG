#!/usr/bin/python3
import argparse

import torch
from mmengine import Config
from mmengine.runner import Runner

""" patch for mmengin model load """
orig_load = torch.load


def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return orig_load(*args, **kwargs)


torch.load = patched_load


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/graph_sd.py",
        help="path to config file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # config
    config = Config.fromfile(args.config)

    # runner
    runner = Runner.from_cfg(config)

    # debug run, just to be sure
    runner.val()

    # train
    runner.train()
