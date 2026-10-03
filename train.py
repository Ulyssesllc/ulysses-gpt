import argparse
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_gpt.dataset import BPETokenizer, TextDataset, clean_hf_dataset, clean_text
from mini_gpt.generate import generate
from mini_gpt.model import GPTConfig, MiniGPT
from mini_gpt.post_train import DPODataset, DPOTrainer, SFTDataset
from mini_gpt.trainer import Trainer, TrainerConfig

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
BLOCK_SIZE = 128


def load_text_dataset(dataset_path: str | None, dataset_url: str | None) -> str:
    """Load a UTF-8 text corpus from a local file or URL."""
    if dataset_url:
        from urllib.parse import urlparse

        filename = Path(urlparse(dataset_url).path).name or "dataset.txt"
        destination = Path(dataset_path) if dataset_path else Path("data") / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            print(f" Downloading dataset from {dataset_url} to {destination}...")
            urllib.request.urlretrieve(dataset_url, destination)
        path = destination
    else:
        path = Path(dataset_path) if dataset_path else Path("tinyshakespeare.txt")
        if dataset_path is None and not path.exists():
            print(" Downloading default TinyShakespeare dataset...")
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(DEFAULT_DATA_URL, path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not UTF-8 text. Convert it to a .txt corpus first."
        ) from exc
    text = clean_text(text)
    if not text:
        raise ValueError(f"Dataset file is empty: {path}")
    print(f" Using dataset: {path}")
    return text


def load_huggingface_text_dataset(
    dataset_name: str,
    split: str,
    text_column: str | None,
    min_characters: int,
    dataset_config: str | None = None,
) -> str:
    """Load, clean, and combine text rows from a Hugging Face dataset."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, name=dataset_config, split=split)
    if text_column is None or text_column == "auto":
        # Prefer conventional prose fields over metadata and identifiers.
        preferred = (
            "text",
            "content",
            "document",
            "article",
            "messages",
            "conversation",
            "data",
            "output",
            "response",
            "prompt",
        )
        text_column = next(
            (name for name in preferred if name in dataset.column_names), None
        )
        if text_column is None:
            # Fall back to the first string or nested field based on dataset features.
            text_column = next(
                (
                    name
                    for name, feature in dataset.features.items()
                    if getattr(feature, "dtype", None) == "string"
                    or getattr(feature, "feature", None) is not None
                ),
                None,
            )
        if text_column is None:
            raise ValueError(
                "Could not infer a text column. Available columns: "
                f"{dataset.column_names}. Pass --text-column COLUMN explicitly."
            )
        print(f" Auto-selected Hugging Face text column: {text_column!r}")
    cleaned = clean_hf_dataset(dataset, text_column, min_characters)
    text = "\n\n".join(cleaned[text_column])
    if not text:
        raise ValueError("No text rows remain after data cleaning.")
    print(
        f" Using Hugging Face dataset: {dataset_name} ({split}), "
        f"{len(cleaned):,} cleaned rows"
    )
    return text


def main():
    parser = argparse.ArgumentParser(description="Train MiniGPT on a text corpus.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset",
        help="Path to a local UTF-8 .txt corpus (default: tinyshakespeare.txt).",
    )
    source.add_argument("--dataset-url", help="URL of a UTF-8 text corpus to download.")
    source.add_argument(
        "--hf-dataset",
        help="Hugging Face dataset repository ID (for example: wikitext).",
    )
    parser.add_argument(
        "--hf-config", help="Optional Hugging Face dataset config name."
    )
    parser.add_argument(
        "--hf-split", default="train", help="Dataset split (default: train)."
    )
    parser.add_argument(
        "--text-column",
        default="auto",
        help="Text column name (default: auto-detect; use 'auto' to enable detection).",
    )
    parser.add_argument(
        "--min-text-characters",
        type=int,
        default=1,
        help="Discard cleaned Hugging Face rows shorter than this value.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Running on device: {device.upper()}")

    if args.hf_dataset:
        raw_text = load_huggingface_text_dataset(
            args.hf_dataset,
            args.hf_split,
            None if args.text_column == "auto" else args.text_column,
            args.min_text_characters,
            args.hf_config,
        )
    else:
        if args.hf_config:
            parser.error("--hf-config requires --hf-dataset")
        raw_text = load_text_dataset(args.dataset, args.dataset_url)

    tokenizer = BPETokenizer("gpt2")
    encoded_tokens = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(encoded_tokens, dtype=torch.long)
    min_tokens = 10 * (BLOCK_SIZE + 1)
    if len(data_tensor) < min_tokens:
        raise ValueError(
            "Dataset is too small after tokenization. It must contain at least "
            f"{min_tokens} tokens so the 90/10 split can create training and "
            "validation samples."
        )
    print(f" Total characters: {len(raw_text):,}")
    print(
        f" Total BPE tokens: {len(data_tensor):,} "
        f"(Compression ratio: {len(raw_text) / len(data_tensor):.2f}x)"
    )

    n_split = int(0.9 * len(data_tensor))
    train_data, val_data = data_tensor[:n_split], data_tensor[n_split:]

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
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
