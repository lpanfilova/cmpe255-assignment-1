import argparse
from model import train

parser = argparse.ArgumentParser(description="Train the NYC taxi trip-duration model")
parser.add_argument("--csv", help="Optional Kaggle train.csv path")
parser.add_argument("--rows", type=int, default=6000, help="Maximum rows to use")
args = parser.parse_args()

if __name__ == "__main__":
    results = train(args.csv, args.rows)
    print("Training complete")
    for name, value in results.items():
        print(f"  {name}: {value}")

