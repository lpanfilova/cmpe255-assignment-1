"""Flask dashboard backed by the reproducible forecast artifact."""

import json

from flask import Flask, jsonify, render_template

from forecasting.pipeline import ARTIFACT_PATH, run_pipeline

app = Flask(__name__)


def results() -> dict:
    if not ARTIFACT_PATH.exists():
        return run_pipeline()
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


@app.get("/")
def dashboard():
    return render_template("index.html", results=results())


@app.get("/api/forecast")
def api_forecast():
    return jsonify(results())


@app.get("/health")
def health():
    payload = results()
    return {"status": payload["operations"]["status"], "artifact": ARTIFACT_PATH.exists(), "champion": payload["champion"]}


if __name__ == "__main__":
    app.run(debug=True, port=5012)
