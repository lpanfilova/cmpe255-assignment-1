"""Flask administration dashboard for association-rule research."""
import csv
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent

def create_app(artifact_dir=None):
    app = Flask(__name__)
    base = Path(artifact_dir or ROOT / "artifacts")
    def read_artifacts():
        if not (base / "metrics.json").exists() or not (base / "rules.csv").exists():
            raise FileNotFoundError("Artifacts missing. Run: python train.py")
        metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
        with (base / "rules.csv").open(encoding="utf-8", newline="") as handle:
            rules = list(csv.DictReader(handle))
        for rule in rules:
            for key in ["support", "confidence", "lift", "leverage", "conviction"]:
                rule[key] = float(rule[key]) if rule[key] else None
        return metrics, rules
    @app.get("/")
    def index(): return render_template("index.html")
    @app.get("/api/health")
    def health(): return jsonify({"status": "ok", "artifacts_ready": (base / "metrics.json").exists()})
    @app.get("/api/dashboard")
    def dashboard():
        try: metrics, rules = read_artifacts()
        except FileNotFoundError as exc: return jsonify({"error": str(exc)}), 503
        min_lift = max(0.0, float(request.args.get("min_lift", 0)))
        query = request.args.get("item", "").strip().lower()
        rules = [r for r in rules if r["lift"] >= min_lift and (not query or query in (r["antecedent"] + " " + r["consequent"]).lower())]
        return jsonify({"metrics": metrics, "rules": rules[:100], "filtered_rules": len(rules)})
    return app

app = create_app()
if __name__ == "__main__": app.run(host="127.0.0.1", port=5003, debug=False)
