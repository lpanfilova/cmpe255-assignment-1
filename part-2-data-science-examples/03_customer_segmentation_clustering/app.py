"""Flask data-science admin dashboard for customer segmentation."""

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent


def create_app(artifact_dir=None):
    app = Flask(__name__)
    base = Path(artifact_dir or ROOT / "artifacts")

    def read_artifacts():
        metrics_path, customers_path = base / "metrics.json", base / "customers_segmented.csv"
        if not metrics_path.exists() or not customers_path.exists():
            raise FileNotFoundError("Artifacts missing. Run: python train.py")
        return json.loads(metrics_path.read_text(encoding="utf-8")), pd.read_csv(customers_path)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "artifacts_ready": (base / "metrics.json").exists()})

    @app.get("/api/dashboard")
    def dashboard():
        try:
            metrics, customers = read_artifacts()
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 503
        gender = request.args.get("gender", "All")
        if gender in {"Male", "Female"}:
            customers = customers[customers["Gender"] == gender]
        points = customers[["CustomerID", "Gender", "Age", "Annual Income (k$)",
                            "Spending Score (1-100)", "cluster", "segment", "silhouette", "pca_x", "pca_y"]]
        profiles = customers.groupby(["cluster", "segment"], as_index=False).agg(
            customers=("CustomerID", "size"), age=("Age", "mean"), income=("Annual Income (k$)", "mean"),
            spending=("Spending Score (1-100)", "mean"), silhouette=("silhouette", "mean"))
        return jsonify({"metrics": metrics, "profiles": profiles.round(3).to_dict("records"),
                        "points": points.round(4).to_dict("records"), "filtered_rows": len(customers)})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
