"""Read-only data-science administration dashboard and inference API."""
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from model import TASKS, load_predictor

ROOT = Path(__file__).resolve().parent


def create_app(artifact_dir=None):
    app = Flask(__name__)
    base = Path(artifact_dir or ROOT / "artifacts")

    def metrics():
        path = base / "metrics.json"
        if not path.exists(): raise FileNotFoundError("Artifacts missing. Run: python train.py")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/")
    def index(): return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "artifacts_ready": (base / "metrics.json").exists(), "tasks": list(TASKS)})

    @app.get("/api/dashboard")
    def dashboard():
        try: return jsonify(metrics())
        except FileNotFoundError as exc: return jsonify({"error": str(exc)}), 503

    @app.post("/api/predict/<task>")
    def predict(task):
        if task not in TASKS: return jsonify({"error": "Unknown task"}), 404
        rows = request.get_json(silent=True)
        if isinstance(rows, dict): rows = [rows]
        if not isinstance(rows, list) or not rows: return jsonify({"error": "Send a JSON object or non-empty list"}), 400
        try:
            import pandas as pd
            predictor = load_predictor(task, base)
            frame = pd.DataFrame(rows)
            prediction = predictor.predict(frame).tolist()
            response = {"task": task, "predictions": prediction, "rows": len(frame)}
            if TASKS[task].problem_type != "regression":
                response["probabilities"] = predictor.predict_proba(frame).to_dict(orient="records")
            return jsonify(response)
        except FileNotFoundError as exc: return jsonify({"error": str(exc)}), 503
        except Exception as exc: return jsonify({"error": f"Invalid inference payload: {exc}"}), 400
    return app


app = create_app()
if __name__ == "__main__": app.run(host="127.0.0.1", port=5007, debug=False)
