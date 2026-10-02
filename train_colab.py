import os
import urllib.request

import torch
from torch.utils.data import DataLoader

from mini_gpt.dataset import BPETokenizer, TextDataset
from mini_gpt.generate import generate
from mini_gpt.model import GPTConfig, MiniGPT
from mini_gpt.post_train import DPODataset, DPOTrainer, SFTDataset
from mini_gpt.trainer import Trainer, TrainerConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Running on device: {device.upper()}")

    data_url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt"
    )
    data_file = "tinyshakespeare.txt"
    if not os.path.exists(data_file):
        print(" Downloading TinyShakespeare dataset...")
        urllib.request.urlretrieve(data_url, data_file)

    with open(data_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = BPETokenizer("gpt2")
    encoded_tokens = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(encoded_tokens, dtype=torch.long)
    print(f" Total characters: {len(raw_text):,}")
    print(
        f" Total BPE tokens: {len(data_tensor):,} "
        f"(Compression ratio: {len(raw_text) / len(data_tensor):.2f}x)"
    )

    n_split = int(0.9 * len(data_tensor))
    train_data, val_data = data_tensor[:n_split], data_tensor[n_split:]

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
    )
    model = MiniGPT(model_config)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f" MiniGPT model size: {num_params:.2f}M parameters")

    train_ds = TextDataset(train_data, model_config.block_size)
    val_ds = TextDataset(val_data, model_config.block_size)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    trainer_config = TrainerConfig(
        max_epochs=2,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=50,
        max_steps=len(train_loader) * 2,
        grad_accum_steps=2,
        device=device,
        checkpoint_dir="checkpoints",
        save_every_steps=100,
    )
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config=trainer_config,
    )
    print("\n === STAGE 1: PRE-TRAINING ===")

    for epoch in range(1, trainer_config.max_epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss, ppl = trainer.evaluate()
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}"
        )

    pretrained_path = trainer.save_checkpoint("pretrained_minigpt.pt")
    print(f" Pre-trained checkpoint saved at: {pretrained_path}")

    print("\n === STAGE 2: SUPERVISED FINE-TUNING (SFT) ===")
    sft_prompts = [
        "ROMEO:\nWhat light through yonder window breaks?",
        "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?",
        "KING RICHARD:\nA horse! a horse! my kingdom\nfor a horse!",
    ]
    sft_responses = [
        "\nIt is the east, and Juliet is the sun.",
        "\nDeny thy father and refuse thy name.",
        "\nSlave, I have set my life upon a cast.",
    ]
    sft_dataset = SFTDataset(
        sft_prompts,
        sft_responses,
        tokenizer,
        model_config.block_size,
    )
    sft_loader = DataLoader(sft_dataset, batch_size=2, shuffle=True)
    sft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(1, 3):
        total_sft_loss = 0.0
        for x_sft, y_sft in sft_loader:
            x_sft, y_sft = x_sft.to(device), y_sft.to(device)
            _, loss = model(x_sft, y_sft)
            sft_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sft_optimizer.step()
            total_sft_loss += loss.item()

        print(
            f"SFT Epoch {epoch} | Average Loss: {total_sft_loss / len(sft_loader):.4f}"
        )

    print("\n === STAGE 3: DIRECT PREFERENCE OPTIMIZATION (DPO) ===")
    dpo_prompts = ["ROMEO:\nShall I hear more?"]
    dpo_chosen = ["\nOr shall I speak at this?"]
    dpo_rejected = ["\nI do not know what to say."]
    dpo_dataset = DPODataset(
        dpo_prompts,
        dpo_chosen,
        dpo_rejected,
        tokenizer,
        model_config.block_size,
    )
    dpo_loader = DataLoader(dpo_dataset, batch_size=1)
    dpo_trainer = DPOTrainer(
        policy_model=model,
        beta=0.1,
        lr=1e-5,
        device=device,
    )

    for step, batch in enumerate(dpo_loader, start=1):
        metrics = dpo_trainer.train_step(batch)
        print(
            f"DPO Step {step} | Loss: {metrics['dpo_loss']:.4f} | "
            f"Chosen Reward: {metrics['chosen_reward']:.4f} | "
            f"Margin: {metrics['reward_margin']:.4f}"
        )

    print("\n === GENERATED TEXT WITH TEMPERATURE & TOP-K ===")
    prompt = "KING HENRY:"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )
    out_ids = generate(
        model,
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=10,
    )
    generated_text = tokenizer.decode(out_ids[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
