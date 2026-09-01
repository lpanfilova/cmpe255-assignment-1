"""FlowForge Flask application."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from flowforge.engine import WorkflowError, run, validate
from flowforge.repository import Repository


ROOT = Path(__file__).resolve().parent


def create_app(database_path: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    repository = Repository(database_path or ROOT / "flowforge.db")
    app.config["repository"] = repository

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "engine": "local-deterministic"})

    @app.get("/api/workflows")
    def workflows():
        return jsonify(repository.list_workflows())

    @app.get("/api/workflows/<workflow_id>")
    def workflow(workflow_id: str):
        found = repository.get_workflow(workflow_id)
        return (jsonify(found), 200) if found else (jsonify({"error": "Workflow not found."}), 404)

    @app.post("/api/validate")
    def validate_workflow():
        plan = validate(_body())
        return jsonify({"valid": True, "plan": plan.order, "levels": plan.levels})

    @app.post("/api/workflows")
    def save_workflow():
        body = _body()
        validate(body)
        return jsonify(repository.save_workflow(body)), 201

    @app.post("/api/workflows/<workflow_id>/runs")
    def run_workflow(workflow_id: str):
        found = repository.get_workflow(workflow_id)
        if not found:
            return jsonify({"error": "Workflow not found."}), 404
        return jsonify(repository.save_run(workflow_id, run(found))), 201

    @app.get("/api/workflows/<workflow_id>/runs")
    def runs(workflow_id: str):
        return jsonify(repository.list_runs(workflow_id))

    @app.errorhandler(WorkflowError)
    def workflow_error(error: WorkflowError):
        return jsonify({"error": str(error)}), 400

    return app


def _body() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise WorkflowError("Request body must be a JSON object.")
    return body


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
