import argparse
import time
from dataclasses import dataclass

import torch
import tyro
from loguru import logger


@dataclass
class Args:
    lr: float = 0.1
    n_layers: int = 4
    save: bool = True


def main():
    args = tyro.cli(Args)
    if args.lr < 0:
        raise ValueError("Learning rate must be positive")

    print(f"Training with learning rate {args.lr} and {args.n_layers} layers")
    # logger.add("log_{time}.log")
    logger.info(f"Training with learning rate {args.lr} and {args.n_layers} layers")
    x = torch.rand(10000, 100000).cuda()
    time.sleep(args.n_layers * 2)
    print("Done!")
    logger.info("Done!")


if __name__ == "__main__":
    main()
