"""Flask chatbot and data-science administration dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request
import torch

from model import WordTokenizer, load_checkpoint


ROOT = Path(__file__).resolve().parent


def create_app(checkpoint_path=None, metrics_path=None):
    app = Flask(__name__)
    app.config.update(
        CHECKPOINT_PATH=Path(checkpoint_path or os.getenv("NANO_LLM_CHECKPOINT", ROOT / "artifacts" / "nano_llm.pt")),
        METRICS_PATH=Path(metrics_path or ROOT / "artifacts" / "metrics.json"),
    )
    state = {"model": None, "metadata": {}, "device": None, "tokenizer": None}
    lock = Lock()

    def ensure_model():
        if state["model"] is None:
            with lock:
                if state["model"] is None:
                    if not app.config["CHECKPOINT_PATH"].exists():
                        raise FileNotFoundError("Checkpoint missing. Run: python train.py")
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    state["model"], state["metadata"] = load_checkpoint(app.config["CHECKPOINT_PATH"], device)
                    state["device"] = device
                    state["tokenizer"] = WordTokenizer(state["metadata"]["tokenizer_vocabulary"])
        return state["model"], state["device"], state["tokenizer"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "checkpoint_ready": app.config["CHECKPOINT_PATH"].exists()})

    @app.get("/api/metrics")
    def metrics():
        if not app.config["METRICS_PATH"].exists():
            return jsonify({"error": "Metrics missing. Run python train.py."}), 404
        return jsonify(json.loads(app.config["METRICS_PATH"].read_text(encoding="utf-8")))

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt or len(prompt) > 1000:
            return jsonify({"error": "Prompt must contain 1–1000 characters."}), 400
        try:
            max_tokens = max(1, min(int(payload.get("max_tokens", 120)), 300))
            temperature = max(0.1, min(float(payload.get("temperature", 0.55)), 1.5))
            model, device, tokenizer = ensure_model()
            formatted = f"\n### dialogue\nUser: {prompt}\nAssistant:"
            encoded = tokenizer.encode(formatted)
            tokens = torch.tensor([encoded], dtype=torch.long, device=device)
            # Greedy decoding is markedly more reliable for this deliberately tiny corpus.
            top_k = 1 if temperature <= 0.7 else 8
            output = model.generate(tokens, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
            generated = tokenizer.decode(output[0, len(encoded) :].tolist())
            answer = generated.split("\nUser:", 1)[0].split("\n#", 1)[0].strip()
            return jsonify({"response": answer, "generated_tokens": max_tokens, "device": str(device)})
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 503

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
