"""Dependency-free concurrent API smoke/load demonstration."""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PAYLOAD = {"distance_miles": 4.2, "passenger_count": 2, "pickup_hour": 17, "day_of_week": 4,
           "temperature_f": 71, "rain": 0, "pickup_zone": "Midtown"}


def hit(url):
    started = time.perf_counter()
    req = urllib.request.Request(url, json.dumps(PAYLOAD).encode(), {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as response:
        ok = response.status == 200
    return ok, (time.perf_counter() - started) * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--requests", type=int, default=50); parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); url = "http://127.0.0.1:5013/api/v1/predict"
    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda _: hit(url), range(args.requests)))
    elapsed = time.perf_counter() - began; latencies = sorted(x[1] for x in results)
    print(json.dumps({"requests": args.requests, "success": sum(x[0] for x in results), "rps": round(args.requests / elapsed, 1),
                      "p50_ms": round(latencies[len(latencies)//2], 1), "p95_ms": round(latencies[int(len(latencies)*.95)-1], 1)}, indent=2))

