"""Association-rule pipeline and deterministic autoresearch search."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Groceries_dataset.csv"


@dataclass(frozen=True)
class Config:
    min_support: float
    min_confidence: float
    min_lift: float
    max_len: int


def load_baskets(path=DATA_PATH, max_baskets=None):
    """Load and validate member-day baskets from the Kaggle Groceries schema."""
    path = Path(path)
    grouped: dict[tuple[str, str], set[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"Member_number", "Date", "itemDescription"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected columns {sorted(expected)}")
        for row in reader:
            member, date, item = (row[key].strip() for key in ("Member_number", "Date", "itemDescription"))
            if member and date and item:
                grouped.setdefault((member, date), set()).add(item.lower())
    baskets = [(f"{member}-{date}", tuple(sorted(items))) for (member, date), items in sorted(grouped.items())]
    if max_baskets:
        baskets = baskets[: int(max_baskets)]
    if not baskets:
        raise ValueError("No valid baskets found")
    return baskets


def count_itemsets(baskets, floor_support=0.01, max_len=3):
    """Apriori-style support counting with downward-closure pruning."""
    n = len(baskets)
    minimum = max(1, math.ceil(floor_support * n))
    counts: dict[tuple[str, ...], int] = Counter((item,) for _, basket in baskets for item in basket)
    frequent_prev = {key for key, value in counts.items() if value >= minimum}
    for size in range(2, max_len + 1):
        candidates = set()
        frequent_items = {x[0] for x in frequent_prev} if size == 2 else None
        for _, basket in baskets:
            eligible = sorted(set(basket) & frequent_items) if size == 2 else sorted(basket)
            for combo in combinations(eligible, size):
                if size == 2 or all(tuple(x for x in combo if x != removed) in frequent_prev for removed in combo):
                    candidates.add(combo)
        level = Counter()
        candidate_lookup = candidates
        for _, basket in baskets:
            for combo in combinations(sorted(basket), size):
                if combo in candidate_lookup:
                    level[combo] += 1
        frequent_prev = {key for key, value in level.items() if value >= minimum}
        counts.update({key: level[key] for key in frequent_prev})
        if not frequent_prev:
            break
    return dict(counts)


def rules_for_config(counts, basket_count, config: Config):
    rules = []
    support = {itemset: count / basket_count for itemset, count in counts.items()}
    for itemset, joint_support in support.items():
        if len(itemset) < 2 or len(itemset) > config.max_len or joint_support < config.min_support:
            continue
        itemset_set = set(itemset)
        for width in range(1, len(itemset)):
            for antecedent in combinations(itemset, width):
                consequent = tuple(sorted(itemset_set - set(antecedent)))
                antecedent = tuple(sorted(antecedent))
                if antecedent not in support or consequent not in support:
                    continue
                confidence = joint_support / support[antecedent]
                lift = confidence / support[consequent]
                leverage = joint_support - support[antecedent] * support[consequent]
                conviction = (1 - support[consequent]) / (1 - confidence) if confidence < 1 else None
                if confidence >= config.min_confidence and lift >= config.min_lift:
                    rules.append({
                        "antecedent": list(antecedent), "consequent": list(consequent),
                        "support": round(joint_support, 6), "confidence": round(confidence, 6),
                        "lift": round(lift, 6), "leverage": round(leverage, 6),
                        "conviction": round(conviction, 6) if conviction is not None else None,
                    })
    return sorted(rules, key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)


def evaluate(rules, baskets, config):
    items = set()
    for rule in rules:
        items.update(rule["antecedent"] + rule["consequent"])
    mean_lift = sum(r["lift"] for r in rules) / len(rules) if rules else 0
    mean_confidence = sum(r["confidence"] for r in rules) / len(rules) if rules else 0
    # Bounded mass of observed rule opportunities. Directional variants share
    # an itemset, so divide summed rule support by two to limit double-counting.
    coverage = min(sum(r["support"] for r in rules) / 2, 1)
    diversity = len(items) / max(1, len({item for _, basket in baskets for item in basket}))
    volume = min(len(rules), 75) / 75
    objective = .30 * min(mean_lift / 4, 1) + .25 * mean_confidence + .20 * coverage + .15 * diversity + .10 * volume
    return {"objective": round(objective, 6), "rule_count": len(rules), "mean_lift": round(mean_lift, 6),
            "mean_confidence": round(mean_confidence, 6), "opportunity_coverage": round(coverage, 6),
            "catalog_coverage": round(diversity, 6), "config": asdict(config)}


def neighbors(config):
    supports, confidences, lifts, lengths = [.001, .0025, .005, .0075, .01], [.02, .04, .06, .08, .1], [1.0, 1.05, 1.1, 1.2, 1.3], [2, 3]
    values = [supports, confidences, lifts, lengths]
    current = [config.min_support, config.min_confidence, config.min_lift, config.max_len]
    result = []
    for axis, options in enumerate(values):
        pos = options.index(current[axis])
        for next_pos in (pos - 1, pos + 1):
            if 0 <= next_pos < len(options):
                candidate = current.copy(); candidate[axis] = options[next_pos]
                result.append(Config(*candidate))
    return result


def train(data_path=DATA_PATH, artifact_dir=None, max_baskets=None):
    started = time.perf_counter()
    baskets = load_baskets(data_path, max_baskets=max_baskets)
    counts = count_itemsets(baskets, floor_support=.001, max_len=3)
    cache = {}
    def score(config):
        key = asdict(config).__repr__()
        if key not in cache:
            rules = rules_for_config(counts, len(baskets), config)
            cache[key] = (evaluate(rules, baskets, config), rules)
        return cache[key]
    current = Config(.005, .06, 1.1, 2)
    path = []
    while True:
        candidates = [current] + neighbors(current)
        ranked = sorted(((score(c)[0]["objective"], c) for c in candidates), reverse=True, key=lambda x: x[0])
        best_score, best = ranked[0]
        path.append({"iteration": len(path), **score(current)[0]})
        if best == current or best_score <= score(current)[0]["objective"]:
            break
        current = best
    # Small exhaustive audit proves whether the hill climb reached the search-space optimum.
    grid = [Config(s, c, l, m) for s in [.001, .0025, .005, .0075, .01] for c in [.02, .04, .06, .08, .1]
            for l in [1.0, 1.05, 1.1, 1.2, 1.3] for m in [2, 3]]
    experiments = sorted((score(c)[0] for c in grid), key=lambda x: x["objective"], reverse=True)
    winner_metrics, winner_rules = score(current)
    global_best = experiments[0]
    top_items = Counter(item for _, basket in baskets for item in basket)
    metrics = {
        "dataset": {"rows": sum(len(b) for _, b in baskets), "baskets": len(baskets), "items": len(top_items),
                    "average_basket_size": round(sum(len(b) for _, b in baskets) / len(baskets), 3),
                    "duplicate_member_day_items_removed": True},
        "winner": winner_metrics,
        "research": {"method": "deterministic coordinate hill climbing", "path": path,
                     "evaluated_configurations": len(cache), "audit_space": len(grid),
                     "global_best_objective": global_best["objective"],
                     "global_optimum_reached": winner_metrics["objective"] == global_best["objective"]},
        "top_items": [{"item": item, "baskets": count, "support": round(count / len(baskets), 6)} for item, count in top_items.most_common(12)],
        "experiments": experiments[:30], "runtime_seconds": round(time.perf_counter() - started, 3),
        "seed": 42, "data_source": "Kaggle Groceries dataset; grouped by Member_number + Date",
    }
    out = Path(artifact_dir or ROOT / "artifacts"); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "rules.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["antecedent", "consequent", "support", "confidence", "lift", "leverage", "conviction"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for rule in winner_rules:
            writer.writerow({**rule, "antecedent": " | ".join(rule["antecedent"]), "consequent": " | ".join(rule["consequent"])})
    return metrics
