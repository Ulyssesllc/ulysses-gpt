import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MiniGPT


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 500


def get_cosine_lr(
    it: int,
    warmup_steps: int,
    max_steps: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    if it < warmup_steps:
        return learning_rate * (it + 1) / (warmup_steps + 1)

    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


class Trainer:
    def __init__(
        self,
        model: MiniGPT,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: Optional[DataLoader[tuple[torch.Tensor, torch.Tensor]]] = None,
        config: Optional[TrainerConfig] = None,
    ):
        self.config = config or TrainerConfig()
        self.device = self.config.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self.optimizer = self.configure_optimizer()

        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
            self.device_type,
            enabled=(self.device_type == "cuda"),
        )
        self.global_step = 0

    def configure_optimizer(self) -> torch.optim.Optimizer:
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        batches_seen = 0

        for step, (x, y) in enumerate(pbar):
            if self.global_step >= self.config.max_steps:
                break
            x, y = x.to(self.device), y.to(self.device)

            lr = get_cosine_lr(
                self.global_step,
                self.config.warmup_steps,
                self.config.max_steps,
                self.config.learning_rate,
                self.config.min_lr,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            with torch.amp.autocast(  # type: ignore[attr-defined]
                self.device_type,
                enabled=(self.device_type == "cuda"),
            ):
                _, loss = self.model(x, y)
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.config.save_every_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            current_loss = loss.item() * self.config.grad_accum_steps
            total_loss += current_loss
            batches_seen += 1
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{lr:.2e}"})

        avg_loss = total_loss / max(1, batches_seen)
        return avg_loss

    @torch.no_grad()
    def evaluate_metrics(self) -> dict[str, float]:
        if not self.val_loader:
            return {"loss": 0.0, "perplexity": float("inf"), "accuracy": 0.0}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(  # type: ignore[attr-defined]
                self.device_type,
                enabled=(self.device_type == "cuda"),
            ):
                logits, loss = self.model(x, y)
            token_count = y.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count
            correct_tokens += (logits.argmax(dim=-1) == y).sum().item()

        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        accuracy = correct_tokens / max(1, total_tokens)
        return {"loss": avg_loss, "perplexity": perplexity, "accuracy": accuracy}

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Return validation loss and perplexity (kept for API compatibility)."""
        metrics = self.evaluate_metrics()
        return metrics["loss"], metrics["perplexity"]

    def save_checkpoint(self, filename: str) -> str:
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
            "model_config": self.model.config,
        }
        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(
            filepath,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        print(f"Checkpoint loaded successfully at step {self.global_step}")
