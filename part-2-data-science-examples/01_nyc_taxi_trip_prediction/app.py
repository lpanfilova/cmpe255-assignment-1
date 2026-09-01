from __future__ import annotations

import json
from flask import Flask, jsonify, render_template, request
import joblib

from model import METRICS_PATH, MODEL_PATH, predict_duration, train


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = testing
    if not MODEL_PATH.exists():
        train(sample_rows=1500 if testing else 6000)
    bundle = joblib.load(MODEL_PATH)

    @app.get("/")
    def index():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return render_template("index.html", metrics=metrics)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=True)

    @app.post("/predict")
    def predict():
        try:
            payload = request.get_json(force=True)
            required = ("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude", "pickup_datetime")
            if any(key not in payload for key in required):
                return jsonify(error="Pickup, drop-off, and date/time are required."), 400
            coords = [float(payload[key]) for key in required[:4]]
            if not (40.4 <= coords[0] <= 41.1 and -74.4 <= coords[1] <= -73.4 and 40.4 <= coords[2] <= 41.1 and -74.4 <= coords[3] <= -73.4):
                return jsonify(error="Choose pickup and drop-off points in the NYC area."), 400
            seconds, distance = predict_duration(payload, bundle)
            return jsonify(
                duration_seconds=round(seconds),
                duration_minutes=round(seconds / 60, 1),
                straight_line_km=round(distance, 2),
                estimated_route_km=round(distance * 1.28, 2),
            )
        except (TypeError, ValueError, KeyError) as exc:
            return jsonify(error=f"Invalid request: {exc}"), 400

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)

