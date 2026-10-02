import pytest
import torch

from mini_gpt.dataset import CharTokenizer
from mini_gpt.model import GPTConfig, MiniGPT
from mini_gpt.post_train import (
    DPODataset,
    DPOTrainer,
    SFTDataset,
    compute_log_probs,
    dpo_loss,
)


@pytest.fixture
def dummy_tokenizer():
    return CharTokenizer("abcdefghijklmnopqrstuvwxyz 0123456789!?\n")


def test_sft_dataset_prompt_masking(dummy_tokenizer):
    prompts = ["abc"]
    responses = ["def"]
    block_size = 16
    ds = SFTDataset(
        prompts,
        responses,
        dummy_tokenizer,
        block_size=block_size,
    )

    assert len(ds) == 1
    input_ids, target_ids = ds[0]
    assert input_ids.shape == (block_size,)
    assert target_ids.shape == (block_size,)

    prompt_len = len(dummy_tokenizer.encode("abc"))
    assert (target_ids[:prompt_len] == -100).all()

    response_len = len(dummy_tokenizer.encode("def"))
    assert (target_ids[prompt_len : prompt_len + response_len] != -100).all()


def test_dpo_dataset_structure(dummy_tokenizer):
    prompts = ["abc"]
    chosen = ["def"]
    rejected = ["xyz"]
    block_size = 16
    ds = DPODataset(
        prompts,
        chosen,
        rejected,
        dummy_tokenizer,
        block_size=block_size,
    )

    assert len(ds) == 1
    sample = ds[0]
    assert "chosen_input_ids" in sample
    assert "chosen_labels" in sample
    assert "rejected_input_ids" in sample
    assert "rejected_labels" in sample
    assert sample["chosen_input_ids"].shape == (block_size,)


def test_compute_log_probs():
    config = GPTConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = MiniGPT(config)
    input_ids = torch.randint(0, 50, (2, 16))
    labels = input_ids.clone()
    labels[:, :4] = -100

    log_probs = compute_log_probs(model, input_ids, labels)
    assert log_probs.shape == (2,)
    assert (log_probs <= 0.0).all()


def test_dpo_loss_computation():
    pi_chosen = torch.tensor([-2.0, -1.5])
    pi_rejected = torch.tensor([-3.0, -4.0])
    ref_chosen = torch.tensor([-2.1, -1.6])
    ref_rejected = torch.tensor([-2.8, -3.5])

    loss, chosen_reward, rejected_reward = dpo_loss(
        pi_chosen,
        pi_rejected,
        ref_chosen,
        ref_rejected,
        beta=0.1,
    )

    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert chosen_reward is not None
    assert rejected_reward is not None


def test_dpo_trainer_train_step(dummy_tokenizer):
    config = GPTConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = MiniGPT(config)
    prompts = ["abc"]
    chosen = ["def"]
    rejected = ["xyz"]
    dpo_ds = DPODataset(
        prompts,
        chosen,
        rejected,
        dummy_tokenizer,
        block_size=16,
    )
    batch = {key: value.unsqueeze(0) for key, value in dpo_ds[0].items()}
    trainer = DPOTrainer(model, beta=0.1, lr=1e-4, device="cpu")
    metrics = trainer.train_step(batch)

    assert "dpo_loss" in metrics
    assert "chosen_reward" in metrics
    assert "rejected_reward" in metrics
    assert "reward_margin" in metrics
    assert metrics["dpo_loss"] > 0.0
