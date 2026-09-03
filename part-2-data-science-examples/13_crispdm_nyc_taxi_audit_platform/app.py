"""Flask auditor website and versioned REST inference surface."""

from flask import Flask, jsonify, render_template, request

from taxi_audit.pipeline import load_or_train, predict

app = Flask(__name__)


@app.get("/")
def dashboard():
    _, report = load_or_train()
    return render_template("index.html", report=report)


@app.post("/api/v1/predict")
def inference():
    try:
        return jsonify(predict(request.get_json(silent=True) or {}))
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/v1/audit")
def audit():
    _, report = load_or_train()
    return jsonify(report)


@app.get("/health")
def health():
    _, report = load_or_train()
    return jsonify({"status": report["mlops"]["status"], "model_version": report["mlops"]["model_version"],
                    "champion": report["champion"], "artifact_hash": report["mlops"]["model_sha256"]})


if __name__ == "__main__":
    app.run(debug=True, port=5013)

