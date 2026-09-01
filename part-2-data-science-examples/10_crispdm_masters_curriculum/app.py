"""Small read-only dashboard over reproducible pipeline artifacts."""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template

from crispdm.pipeline import ARTIFACT_PATH, run_workflows

app = Flask(__name__)


def results():
    if not ARTIFACT_PATH.exists():
        return run_workflows()
    return json.loads(Path(ARTIFACT_PATH).read_text(encoding="utf-8"))


@app.get("/")
def dashboard():
    return render_template("index.html", results=results())


@app.get("/api/results")
def api_results():
    return jsonify(results())


@app.get("/health")
def health():
    return {"status": "ok", "artifact": ARTIFACT_PATH.exists()}


if __name__ == "__main__":
    app.run(debug=True, port=5000)

