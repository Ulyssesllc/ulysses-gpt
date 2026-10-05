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
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if eos_token_id is not None:
            idx_next = torch.where(
                finished[:, None],
                torch.full_like(idx_next, eos_token_id),
                idx_next,
            )
        idx = torch.cat((idx, idx_next), dim=1)
        if eos_token_id is not None:
            finished |= idx_next.squeeze(1) == eos_token_id
            if bool(finished.all()):
                break

    return idx


@torch.no_grad()
def generate_with_metrics(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generate text and report time-to-first-token and aggregate throughput."""
    model.eval()
    if idx.is_cuda:
        torch.cuda.synchronize(idx.device)
    started = time.perf_counter()
    first_token_seconds = 0.0
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    generated_token_count = 0
    for token_index in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("Inf")
        idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
        active = ~finished
        if eos_token_id is not None:
            idx_next = torch.where(
                finished[:, None],
                torch.full_like(idx_next, eos_token_id),
                idx_next,
            )
        idx = torch.cat((idx, idx_next), dim=1)
        generated_token_count += int(active.sum().item())
        if token_index == 0:
            if idx.is_cuda:
                torch.cuda.synchronize(idx.device)
            first_token_seconds = time.perf_counter() - started
        if eos_token_id is not None:
            finished |= idx_next.squeeze(1) == eos_token_id
            if bool(finished.all()):
                break
    if idx.is_cuda:
        torch.cuda.synchronize(idx.device)
    elapsed = time.perf_counter() - started
    generated_tokens = generated_token_count
    return idx, {
        "ttft_seconds": first_token_seconds,
        "throughput_tokens_per_second": generated_tokens / max(elapsed, 1e-12),
        "generation_seconds": elapsed,
        "generated_tokens": float(generated_tokens),
    }
