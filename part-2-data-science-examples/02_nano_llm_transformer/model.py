"""A compact decoder-only Transformer for educational local language modeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import unicodedata
import re

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 96
    block_size: int = 192
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 96
    dropout: float = 0.05
    bias: bool = False

    def validate(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("head dimension must be even for RoPE")


class WordTokenizer:
    """Small corpus-fitted word/punctuation tokenizer persisted with each checkpoint."""

    pattern = re.compile(r"\n|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", re.ASCII)

    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)

    @classmethod
    def fit(cls, text: str) -> "WordTokenizer":
        tokens = cls.pattern.findall(cls._normalize(text))
        return cls(["<unk>"] + sorted(set(tokens)))

    @staticmethod
    def _normalize(text: str) -> str:
        return unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, 0) for token in self.pattern.findall(self._normalize(text))]

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.vocabulary[token] if 0 <= token < self.vocab_size else "<unk>" for token in token_ids]
        text = " ".join(tokens).replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
        text = re.sub(r"\s+([.,!?;:%)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = text.replace(" - ", "-")
        return text


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        length = q.size(-2)
        cos, sin = self.cos[:, :, :length], self.sin[:, :, :length]

        def rotate(x: Tensor) -> Tensor:
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, config.block_size)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.proj(y.transpose(1, 2).contiguous().view(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = 32 * ((hidden + 31) // 32)
        self.gate_up = nn.Linear(config.n_embd, 2 * hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_embd)
        self.ffn_norm = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn = SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class NanoLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if tokens.size(1) > self.config.block_size:
            raise ValueError(f"sequence exceeds block size {self.config.block_size}")
        x = self.dropout(self.token_embedding(tokens))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.inference_mode()
    def generate(
        self, tokens: Tensor, max_new_tokens: int = 120, temperature: float = 0.8, top_k: int = 40
    ) -> Tensor:
        self.eval()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1] / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -torch.inf
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def save_checkpoint(path: str | Path, model: NanoLLM, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "model": model.state_dict(), "metadata": metadata or {}}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[NanoLLM, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model = NanoLLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint.get("metadata", {})
