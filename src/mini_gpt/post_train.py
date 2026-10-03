import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .model import MiniGPT


def _pack_prompt_response(
    prompt_ids: List[int], response_ids: List[int], block_size: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    """Keep the response and the most recent prompt tokens within the context."""
    if block_size < 2:
        raise ValueError("block_size must be at least 2 for prompt/response training")
    response_ids = response_ids[: block_size - 1]
    prompt_ids = prompt_ids[-(block_size - len(response_ids)) :]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    pad = block_size - len(input_ids)
    return input_ids + [pad_token_id] * pad, labels + [-100] * pad


class SFTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        prompts: List[str],
        responses: List[str],
        tokenizer: Any,
        block_size: int,
        pad_token_id: int = 0,
    ):
        self.samples = []

        for prompt, response in zip(prompts, responses):
            p_ids = tokenizer.encode(prompt)
            r_ids = tokenizer.encode(response)

            combined_input, combined_target = _pack_prompt_response(
                p_ids, r_ids, block_size, pad_token_id
            )

            self.samples.append(
                (
                    torch.tensor(combined_input, dtype=torch.long),
                    torch.tensor(combined_target, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class DPODataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        tokenizer: Any,
        block_size: int,
        pad_token_id: int = 0,
    ):
        self.samples = []

        for p, c, r in zip(prompts, chosen_responses, rejected_responses):
            p_ids = tokenizer.encode(p)
            c_ids = tokenizer.encode(c)
            r_ids = tokenizer.encode(r)

            c_input, c_target = _pack_prompt_response(
                p_ids, c_ids, block_size, pad_token_id
            )
            r_input, r_target = _pack_prompt_response(
                p_ids, r_ids, block_size, pad_token_id
            )

            self.samples.append(
                {
                    "chosen_input_ids": torch.tensor(c_input, dtype=torch.long),
                    "chosen_labels": torch.tensor(c_target, dtype=torch.long),
                    "rejected_input_ids": torch.tensor(r_input, dtype=torch.long),
                    "rejected_labels": torch.tensor(r_target, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def compute_log_probs(
    model: MiniGPT,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits, _ = model(input_ids, labels)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)

    loss_mask = shift_labels != -100

    shift_labels_clamped = shift_labels.clone()
    shift_labels_clamped[~loss_mask] = 0

    per_token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels_clamped.unsqueeze(-1),
    ).squeeze(-1)

    return (per_token_log_probs * loss_mask.float()).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class DPOTrainer:
    def __init__(
        self,
        policy_model: MiniGPT,
        beta: float = 0.1,
        lr: float = 5e-6,
        weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.policy_model = policy_model.to(device)
        self.device = device
        self.beta = beta

        self.ref_model = copy.deepcopy(policy_model).to(device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.policy_model.train()

        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        pi_chosen_logps = compute_log_probs(
            self.policy_model, chosen_ids, chosen_labels
        )
        pi_rejected_logps = compute_log_probs(
            self.policy_model, rejected_ids, rejected_labels
        )

        with torch.no_grad():
            ref_chosen_logps = compute_log_probs(
                self.ref_model, chosen_ids, chosen_labels
            )
            ref_rejected_logps = compute_log_probs(
                self.ref_model, rejected_ids, rejected_labels
            )

        loss, chosen_reward, rejected_reward = dpo_loss(
            pi_chosen_logps,
            pi_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta,
        )

        self.optimizer.zero_grad()
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        reward_margin = chosen_reward - rejected_reward

        return {
            "dpo_loss": loss.item(),
            "chosen_reward": chosen_reward.item(),
            "rejected_reward": rejected_reward.item(),
            "reward_margin": reward_margin.item(),
        }
