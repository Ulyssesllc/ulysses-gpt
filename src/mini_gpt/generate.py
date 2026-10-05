import time

import torch
import torch.nn.functional as F

from .model import MiniGPT


@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


@torch.no_grad()
def generate_with_metrics(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generate text and report time-to-first-token and aggregate throughput."""
    model.eval()
    if idx.is_cuda:
        torch.cuda.synchronize(idx.device)
    started = time.perf_counter()
    first_token_seconds = 0.0
    for token_index in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("Inf")
        idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        if token_index == 0:
            if idx.is_cuda:
                torch.cuda.synchronize(idx.device)
            first_token_seconds = time.perf_counter() - started
    if idx.is_cuda:
        torch.cuda.synchronize(idx.device)
    elapsed = time.perf_counter() - started
    generated_tokens = max_new_tokens * idx.size(0)
    return idx, {
        "ttft_seconds": first_token_seconds,
        "throughput_tokens_per_second": generated_tokens / max(elapsed, 1e-12),
        "generation_seconds": elapsed,
        "generated_tokens": float(generated_tokens),
    }
