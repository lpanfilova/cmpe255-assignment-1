"""CLI entry point for bounded AutoGluon training."""
import argparse
from pathlib import Path
from model import train_all

parser = argparse.ArgumentParser()
parser.add_argument("--time-limit", type=int, default=15, help="Seconds per candidate")
parser.add_argument("--max-candidates", type=int, choices=(1, 2, 3), default=2)
parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
args = parser.parse_args()
result = train_all(args.artifact_dir, args.time_limit, args.max_candidates)
print(f"Trained {len(result['tasks'])} tasks in {result['total_duration_sec']:.1f}s")
for key, task in result["tasks"].items():
    print(f"  {key:10} winner={task['winner']:<20} metrics={task['test_metrics']}")
