"""Flask administration dashboard for anomaly research and holdout triage."""
import csv
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent

def create_app(artifact_dir=None):
    app = Flask(__name__)
    base = Path(artifact_dir or ROOT / "artifacts")
    def read_artifacts():
        if not (base / "metrics.json").exists() or not (base / "scored_holdout.csv").exists():
            raise FileNotFoundError("Artifacts missing. Run: python train.py")
        metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
        with (base / "scored_holdout.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["record_id"] = int(row["record_id"]); row["is_anomaly"] = int(row["is_anomaly"])
            row["anomaly_score"] = float(row["anomaly_score"]); row["flagged"] = row["flagged"] == "True"
        return metrics, rows
    @app.get("/")
    def index(): return render_template("index.html")
    @app.get("/api/health")
    def health(): return jsonify({"status": "ok", "artifacts_ready": (base / "metrics.json").exists()})
    @app.get("/api/dashboard")
    def dashboard():
        try: metrics, rows = read_artifacts()
        except FileNotFoundError as exc: return jsonify({"error": str(exc)}), 503
        flagged = request.args.get("flagged", "all")
        if flagged in {"true", "false"}: rows = [r for r in rows if r["flagged"] == (flagged == "true")]
        limit = min(200, max(1, int(request.args.get("limit", 50))))
        return jsonify({"metrics": metrics, "records": rows[:limit], "filtered_records": len(rows)})
    return app

app = create_app()
if __name__ == "__main__": app.run(host="127.0.0.1", port=5005, debug=False)

