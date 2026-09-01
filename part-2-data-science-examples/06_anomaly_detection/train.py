import argparse
from model import DEFAULT_DATA, train

parser = argparse.ArgumentParser(description="Run Annthyroid anomaly autoresearch")
parser.add_argument("--data", default=str(DEFAULT_DATA))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--alert-budget", type=float, default=.05)
args = parser.parse_args()

if __name__ == "__main__":
    result = train(args.data, seed=args.seed, alert_budget=args.alert_budget)
    print("Winner:", result["winner"])

