"""Generate reproducible samples from a trained NanoLLM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import WordTokenizer, load_checkpoint


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = [
    "What is machine learning?",
    "Explain overfitting.",
    "What is a good experiment?",
    "How should I evaluate a model?",
]


def generate_answer(model, tokenizer, prompt: str, seed: int = 7, max_tokens: int = 80) -> str:
    torch.manual_seed(seed)
    prefix = f"\n### dialogue\nUser: {prompt}\nAssistant:"
    encoded = tokenizer.encode(prefix)
    device = next(model.parameters()).device
    output = model.generate(
        torch.tensor([encoded], dtype=torch.long, device=device),
        max_new_tokens=max_tokens,
        temperature=0.5,
        top_k=1,
    )
    text = tokenizer.decode(output[0, len(encoded) :].tolist())
    return text.split("\nUser:", 1)[0].split("\n#", 1)[0].strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "nano_llm.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args(argv)
    model, metadata = load_checkpoint(args.checkpoint)
    tokenizer = WordTokenizer(metadata["tokenizer_vocabulary"])
    print(f"checkpoint={args.checkpoint.name} steps={metadata.get('trained_steps', 'unknown')} seed={args.seed}")
    for prompt in args.prompts:
        print(f"\nPrompt: {prompt}\nGeneration: {generate_answer(model, tokenizer, prompt, args.seed, args.max_tokens)}")


if __name__ == "__main__":
    main()
