#!/usr/bin/python3
import argparse
from pathlib import Path

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
    parser.add_argument(
        "--weight",
        type=str,
        default="gryphgen_graph_sd_mathwriting_epoch_50.pth",
        help="path to weight file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="dataset split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="~/gen",
        help="path to output directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = Path(args.output).joinpath(args.split)

    # config
    config = Config.fromfile(args.config)
    config.test_dataloader.dataset.update(filter_cfg=dict(split=args.split))
    config.test_evaluator = dict(type="DumpData", output_dir=str(output_dir))

    # runner
    runner = Runner.from_cfg(config)
    runner.load_checkpoint(args.weight)
    runner.test()
