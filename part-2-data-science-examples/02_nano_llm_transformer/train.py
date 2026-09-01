"""Train NanoLLM and persist a reproducible checkpoint plus dashboard metrics."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch

from model import ModelConfig, NanoLLM, WordTokenizer, save_checkpoint


ROOT = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokens(path: Path) -> tuple[torch.Tensor, torch.Tensor, WordTokenizer]:
    """Split at dialogue boundaries so validation never contains a severed record."""
    raw_text = path.read_text(encoding="utf-8")
    dialogues = [item.strip() for item in raw_text.split("\n\n") if item.strip()]
    if len(dialogues) < 10:
        raise ValueError("corpus needs at least 10 dialogue records")
    validation_count = max(2, round(len(dialogues) * 0.15))
    boundary = "\n### dialogue\n"
    train_text = boundary + boundary.join(dialogues[:-validation_count]) + "\n"
    validation_text = boundary + boundary.join(dialogues[-validation_count:]) + "\n"
    tokenizer = WordTokenizer.fit(raw_text + boundary)
    return (torch.tensor(tokenizer.encode(train_text), dtype=torch.long),
            torch.tensor(tokenizer.encode(validation_text), dtype=torch.long), tokenizer)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    if len(data) <= block_size:
        raise ValueError(f"corpus split needs more than {block_size} byte tokens")
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.inference_mode()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, batches=5):
    model.eval()
    result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def run_training(config: ModelConfig, args, train_data, val_data, device, steps: int | None = None):
    steps = steps or args.steps
    model = NanoLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
    history = []
    started = time.perf_counter()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    eval_interval = max(1, min(args.eval_interval, steps))

    for step in range(steps):
        lr = learning_rate(step, steps, args.learning_rate, min(args.warmup_steps, steps // 4))
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, args.batch_size, config.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        amp = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with amp:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(optimizer)
        scaler.update()

        if step == 0 or (step + 1) % eval_interval == 0 or step + 1 == steps:
            losses = estimate_loss(
                model, train_data, val_data, args.batch_size, config.block_size, device, args.eval_batches
            )
            history.append({"step": step + 1, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr, "grad_norm": grad_norm})
            print(f"step {step + 1:4d} | train {losses['train']:.3f} | val {losses['val']:.3f} | lr {lr:.2e}")

    return model, history, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "tiny_dialogues.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "nano_llm.pt")
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts" / "metrics.json")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search", action="store_true", help="Hill-climb over three tiny configurations before final training")
    parser.add_argument("--search-steps", type=int, default=40)
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    device = choose_device(args.device)
    train_data, val_data, tokenizer = load_tokens(args.corpus)
    experiments = []
    candidates = [(args.n_layer, args.n_embd)]
    if args.search:
        candidates = [(1, 64), (2, 96), (3, 96)]
        print("Running bounded capacity search...")
        for layers, width in candidates:
            config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_layer=layers, n_head=args.n_head, n_embd=width, dropout=args.dropout)
            seed_everything(args.seed)
            candidate, history, seconds = run_training(config, args, train_data, val_data, device, args.search_steps)
            experiments.append({"name": f"L{layers}-D{width}", "parameters": candidate.parameter_count, "val_loss": history[-1]["val_loss"], "seconds": seconds})
        best = min(experiments, key=lambda item: item["val_loss"])["name"]
        chosen = candidates[[f"L{x}-D{y}" for x, y in candidates].index(best)]
    else:
        chosen = candidates[0]

    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_layer=chosen[0], n_head=args.n_head, n_embd=chosen[1], dropout=args.dropout)
    seed_everything(args.seed)
    model, history, seconds = run_training(config, args, train_data, val_data, device)
    final_val = history[-1]["val_loss"]
    metadata = {"final_val_loss": final_val, "trained_steps": args.steps, "corpus": args.corpus.name, "tokenizer_vocabulary": tokenizer.vocabulary}
    save_checkpoint(args.output, model, metadata)
    metrics = {
        "model": {**asdict(config), "parameters": model.parameter_count, "checkpoint_mb": args.output.stat().st_size / 1_048_576},
        "training": {"device": str(device), "steps": args.steps, "seconds": seconds, "tokens_seen": args.steps * args.batch_size * args.block_size, "seed": args.seed, "history": history},
        "data": {"source": args.corpus.name, "train_tokens": len(train_data), "validation_tokens": len(val_data), "tokenizer": f"Corpus-fitted words/punctuation ({tokenizer.vocab_size} tokens)", "split": "Dialogue-boundary 85/15", "license": "Original educational corpus (repository-owned)"},
        "evaluation": {"final_train_loss": history[-1]["train_loss"], "final_val_loss": final_val, "perplexity": math.exp(min(final_val, 20)), "warning": "Tiny-corpus validation loss measures imitation, not general intelligence or factual accuracy."},
        "experiments": experiments,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {model.parameter_count:,}-parameter model to {args.output}")
    return metrics


if __name__ == "__main__":
    main()
