import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mini_gpt.model import GPTConfig, MiniGPT
from mini_gpt.trainer import Trainer, TrainerConfig, get_cosine_lr


def test_cosine_learning_rate_schedule():
    warmup_steps = 10
    max_steps = 100
    base_lr = 1e-3
    min_lr = 1e-4

    lr_warmup = get_cosine_lr(
        0,
        warmup_steps,
        max_steps,
        base_lr,
        min_lr,
    )
    assert lr_warmup < base_lr

    lr_peak = get_cosine_lr(
        warmup_steps,
        warmup_steps,
        max_steps,
        base_lr,
        min_lr,
    )
    assert pytest.approx(lr_peak, abs=1e-5) == base_lr

    lr_end = get_cosine_lr(
        150,
        warmup_steps,
        max_steps,
        base_lr,
        min_lr,
    )
    assert lr_end == min_lr


def test_trainer_loop_and_checkpoint(tmp_path):
    config = GPTConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = MiniGPT(config)

    data = torch.randint(0, 50, (64, 17))
    ds = TensorDataset(data[:, :-1], data[:, 1:])
    train_loader = DataLoader(ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(ds, batch_size=8)
    trainer_cfg = TrainerConfig(
        max_epochs=1,
        learning_rate=1e-3,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        save_every_steps=2,
    )
    trainer = Trainer(model, train_loader, val_loader, config=trainer_cfg)

    avg_loss = trainer.train_epoch(epoch=1)
    assert avg_loss > 0.0

    val_loss, ppl = trainer.evaluate()
    assert val_loss > 0.0
    assert ppl >= 1.0

    ckpt_file = "model_test.pt"
    ckpt_path = trainer.save_checkpoint(ckpt_file)
    assert os.path.exists(ckpt_path)

    trainer.load_checkpoint(ckpt_path)
    assert trainer.global_step >= 0
