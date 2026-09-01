import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app
from model import ModelConfig, NanoLLM, WordTokenizer, load_checkpoint
from train import main


def test_training_checkpoint_generation_and_app(tmp_path):
    checkpoint = tmp_path / "model.pt"
    metrics = tmp_path / "metrics.json"
    main(["--steps", "2", "--eval-interval", "1", "--eval-batches", "1", "--batch-size", "2", "--block-size", "32", "--n-layer", "1", "--n-head", "2", "--n-embd", "32", "--device", "cpu", "--output", str(checkpoint), "--metrics", str(metrics)])
    assert checkpoint.exists() and json.loads(metrics.read_text())["training"]["steps"] == 2

    model, metadata = load_checkpoint(checkpoint)
    assert metadata["trained_steps"] == 2
    tokenizer = WordTokenizer(metadata["tokenizer_vocabulary"])
    prompt = tokenizer.encode("User: hello\nAssistant:")
    generated = model.generate(torch.tensor([prompt]), max_new_tokens=4, top_k=10)
    assert generated.shape[1] == len(prompt) + 4
    decoded = tokenizer.decode(generated[0].tolist())
    assert isinstance(decoded, str) and "�" not in decoded

    client = create_app(checkpoint, metrics).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json["checkpoint_ready"] is True
    assert client.get("/api/metrics").status_code == 200
    response = client.post("/api/chat", json={"prompt": "hello", "max_tokens": 2})
    assert response.status_code == 200 and isinstance(response.json["response"], str)


def test_model_rejects_invalid_context():
    model = NanoLLM(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32))
    try:
        model(torch.zeros((1, 9), dtype=torch.long))
        assert False, "expected context validation"
    except ValueError:
        pass
