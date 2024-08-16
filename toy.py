import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_layers", type=int, default=4)
    args = parser.parse_args()
    if args.lr < 0:
        raise ValueError("Learning rate must be positive")

    print(f"Training with learning rate {args.lr} and {args.n_layers} layers")
    x = torch.rand(10000, 100000).cuda()
    time.sleep(args.n_layers * 2)
    print("Done!")


if __name__ == "__main__":
    main()
