import argparse

from model import DEFAULT_DATA, train

parser = argparse.ArgumentParser(description="Run customer-segmentation autoresearch")
parser.add_argument("--csv", default=str(DEFAULT_DATA))
parser.add_argument("--stability-rounds", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if __name__ == "__main__":
    result = train(args.csv, seed=args.seed, stability_rounds=args.stability_rounds)
    print("Best configuration:", result["winner"])
