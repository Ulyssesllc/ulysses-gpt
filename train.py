import argparse
import urllib.request
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mini_gpt.dataset import BPETokenizer, TextDataset, clean_text
from mini_gpt.generate import generate
from mini_gpt.model import GPTConfig, MiniGPT
from mini_gpt.post_train import DPODataset, DPOTrainer, SFTDataset
from mini_gpt.trainer import Trainer, TrainerConfig

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def _stringify(value: Any) -> str:
    """Convert strings and common conversation structures to readable text."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        role = value.get("role") or value.get("from")
        content = value.get("content", value.get("value", value.get("text")))
        if content is not None:
            body = _stringify(content)
            return f"{role}: {body}" if isinstance(role, str) else body
        return "\n".join(filter(None, (_stringify(v) for v in value.values())))
    if isinstance(value, (list, tuple)):
        return "\n".join(filter(None, (_stringify(v) for v in value)))
    return ""


def _choose_column(dataset: Any, requested: str, preferred: tuple[str, ...]) -> str:
    if requested != "auto":
        if requested not in dataset.column_names:
            raise ValueError(
                f"Column {requested!r} not found. Available: {dataset.column_names}"
            )
        return requested
    for name in preferred:
        if name in dataset.column_names:
            return name
    for name, feature in dataset.features.items():
        if getattr(feature, "dtype", None) == "string":
            return name
    raise ValueError(
        f"Could not infer a column from {dataset.column_names}; specify it explicitly."
    )


def _load_hf_split(name: str, config: str | None, split: str) -> Any:
    from datasets import load_dataset

    # Hugging Face expects owner/repo IDs. Keep the common legacy shortcut usable.
    if name == "wikitext":
        name = "Salesforce/wikitext"
    elif "/" not in name:
        raise ValueError(
            f"Invalid Hugging Face dataset ID {name!r}. Use the full "
            "'namespace/repository' form (for example 'Salesforce/wikitext')."
        )
    return load_dataset(name, name=config, split=split)


def _texts(dataset: Any, column: str, min_characters: int = 1):
    for row in dataset:
        text = clean_text(_stringify(row[column]))
        if len(text) >= min_characters:
            yield text


def _encode_texts(
    texts: Any, tokenizer: BPETokenizer, max_tokens: int, label: str
) -> torch.Tensor:
    """Encode in batches and cap retained tokens to bound host memory."""
    token_chunks: list[torch.Tensor] = []
    token_count = 0
    print(f" Encoding {label} (limit: {max_tokens:,} tokens)...", flush=True)
    iterator = iter(texts)
    with tqdm(desc=f"Tokenizing {label}", unit="batch") as progress:
        while token_count < max_tokens:
            batch = list(islice(iterator, 256))
            if not batch:
                break
            encoded = tokenizer.encoder.encode_batch(
                batch, allowed_special={"<|endoftext|>"}
            )
            for ids in encoded:
                if not ids:
                    continue
                remaining = max_tokens - token_count
                ids = ids[:remaining]
                if ids:
                    token_chunks.append(torch.tensor(ids, dtype=torch.int32))
                    token_count += len(ids)
                if token_count >= max_tokens:
                    break
            progress.update(1)
    if not token_chunks:
        raise ValueError(f"No tokens found in {label} data.")
    data = torch.cat(token_chunks).long()
    print(f" Encoded {len(data):,} tokens from {label}.", flush=True)
    return data


def _load_local_text(path: str | None, url: str | None) -> list[str]:
    if url:
        from urllib.parse import urlparse

        destination = (
            Path(path)
            if path
            else Path("data") / (Path(urlparse(url).path).name or "dataset.txt")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            print(f"Downloading {url}...", flush=True)
            urllib.request.urlretrieve(url, destination)
        path = str(destination)
    path = path or "tinyshakespeare.txt"
    file_path = Path(path)
    if not file_path.exists() and not url:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DEFAULT_DATA_URL, file_path)
    text = clean_text(file_path.read_text(encoding="utf-8"))
    if not text:
        raise ValueError(f"Dataset file is empty: {file_path}")
    return [text]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MiniGPT on configurable datasets."
    )
    parser.add_argument(
        "--task", choices=("pretrain", "sft", "dpo"), default="pretrain"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", help="Local UTF-8 text file.")
    source.add_argument("--dataset-url", help="URL of a UTF-8 text file.")
    source.add_argument("--hf-dataset", help="Hugging Face dataset repository ID.")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--text-column", default="auto")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--chosen-column", default="chosen")
    parser.add_argument("--rejected-column", default="rejected")
    parser.add_argument("--min-text-characters", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2_000_000,
        help="Maximum tokens retained per pretraining split (bounds Colab RAM).",
    )
    parser.add_argument(
        "--max-val-tokens",
        type=int,
        default=20_000,
        help="Maximum validation tokens encoded per run.",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Maximum SFT/DPO examples to keep in memory.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume")
    parser.add_argument(
        "--checkpoint", help="Pretrained checkpoint required for SFT/DPO."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()

    if args.hf_config and not args.hf_dataset:
        parser.error("--hf-config requires --hf-dataset")
    if args.task != "pretrain" and not args.hf_dataset:
        parser.error("SFT/DPO require --hf-dataset with the relevant columns")
    if args.task != "pretrain" and not args.checkpoint:
        parser.error("SFT/DPO require --checkpoint from a pretrained run")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}", flush=True)
    tokenizer = BPETokenizer("gpt2")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.1,
    )
    saved = None
    init_path = args.checkpoint or args.resume
    if init_path:
        saved = torch.load(init_path, map_location="cpu", weights_only=False)
        config = saved.get("model_config", config)
    model = MiniGPT(config)
    if args.checkpoint or args.resume:
        model.load_state_dict(saved.get("model_state_dict", saved))

    if args.task == "pretrain":
        if args.hf_dataset:
            train_hf = _load_hf_split(args.hf_dataset, args.hf_config, args.hf_split)
            text_column = _choose_column(
                train_hf,
                args.text_column,
                (
                    "text",
                    "content",
                    "document",
                    "article",
                    "messages",
                    "conversation",
                    "data",
                ),
            )
            print(f"Using text column {text_column!r}")
            train_hf = train_hf.shuffle(seed=args.seed)
            train_texts = _texts(train_hf, text_column, args.min_text_characters)
            try:
                valid_hf = _load_hf_split(
                    args.hf_dataset, args.hf_config, args.validation_split
                )
                valid_column = (
                    text_column
                    if text_column in valid_hf.column_names
                    else _choose_column(
                        valid_hf,
                        "auto",
                        (text_column, "text", "content", "document", "article"),
                    )
                )
                valid_texts = _texts(valid_hf, valid_column, args.min_text_characters)
                print(f"Using dataset validation split {args.validation_split!r}")
            except ValueError as exc:
                print(
                    f"Validation split unavailable ({exc}); splitting train data 95/5."
                )
                split = train_hf.train_test_split(test_size=0.05, seed=args.seed)
                train_texts = _texts(
                    split["train"], text_column, args.min_text_characters
                )
                valid_texts = _texts(
                    split["test"], text_column, args.min_text_characters
                )
        else:
            train_texts = _load_local_text(args.dataset, args.dataset_url)
            cut = max(1, int(len(train_texts[0]) * 0.95))
            valid_texts, train_texts = [train_texts[0][cut:]], [train_texts[0][:cut]]

        train_data = _encode_texts(train_texts, tokenizer, args.max_tokens, "train")
        valid_data = _encode_texts(
            valid_texts, tokenizer, args.max_val_tokens, "validation"
        )
        if len(train_data) <= args.block_size or len(valid_data) <= args.block_size:
            raise ValueError(
                "Train and validation splits must each exceed --block-size tokens."
            )
        train_loader = DataLoader(
            TextDataset(train_data, args.block_size),
            batch_size=args.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            TextDataset(valid_data, args.block_size), batch_size=args.batch_size
        )
        total_steps = min(
            args.max_steps,
            max(1, len(train_loader) // args.grad_accum_steps * args.epochs),
        )
        trainer_config = TrainerConfig(
            max_epochs=args.epochs,
            learning_rate=args.learning_rate,
            min_lr=args.learning_rate * 0.1,
            warmup_steps=min(50, total_steps // 10),
            max_steps=total_steps,
            grad_accum_steps=args.grad_accum_steps,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            save_every_steps=10000,
        )
        trainer = Trainer(model, train_loader, valid_loader, trainer_config)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(epoch)
            val_loss, ppl = trainer.evaluate()
            print(
                f"Epoch {epoch}: train={train_loss:.4f}, "
                f"val={val_loss:.4f}, ppl={ppl:.2f}"
            )
            if trainer.global_step >= total_steps:
                break
        trainer.save_checkpoint("final_minigpt.pt")

    else:
        dataset = _load_hf_split(args.hf_dataset, args.hf_config, args.hf_split)
        if args.task == "sft":
            required = (args.prompt_column, args.response_column)
            missing = [col for col in required if col not in dataset.column_names]
            if missing:
                raise ValueError(
                    f"SFT columns missing: {missing}; available: {dataset.column_names}"
                )
            count = min(len(dataset), args.max_samples)
            prompts = [_stringify(dataset[i][args.prompt_column]) for i in range(count)]
            responses = [
                _stringify(dataset[i][args.response_column]) for i in range(count)
            ]
            post_data = SFTDataset(prompts, responses, tokenizer, args.block_size)
            loader = DataLoader(post_data, batch_size=args.batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            model.to(device).train()
            for epoch in range(args.epochs):
                total = 0.0
                for x, y in tqdm(loader, desc=f"SFT epoch {epoch + 1}"):
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    _, loss = model(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total += loss.item()
                print(f"SFT epoch {epoch + 1}: loss={total / max(1, len(loader)):.4f}")
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "model_config": config},
                Path(args.checkpoint_dir) / "sft_minigpt.pt",
            )
        else:
            required = (args.prompt_column, args.chosen_column, args.rejected_column)
            missing = [col for col in required if col not in dataset.column_names]
            if missing:
                raise ValueError(
                    f"DPO columns missing: {missing}; available: {dataset.column_names}"
                )
            count = min(len(dataset), args.max_samples)
            prompts = [_stringify(dataset[i][args.prompt_column]) for i in range(count)]
            chosen = [_stringify(dataset[i][args.chosen_column]) for i in range(count)]
            rejected = [
                _stringify(dataset[i][args.rejected_column]) for i in range(count)
            ]
            post_data = DPODataset(
                prompts, chosen, rejected, tokenizer, args.block_size
            )
            loader = DataLoader(post_data, batch_size=args.batch_size, shuffle=True)
            trainer = DPOTrainer(model, lr=args.learning_rate, device=device)
            for epoch in range(args.epochs):
                for step, batch in enumerate(
                    tqdm(loader, desc=f"DPO epoch {epoch + 1}"), 1
                ):
                    metrics = trainer.train_step(batch)
                    if step % 20 == 0:
                        print(
                            f"DPO step {step}: loss={metrics['dpo_loss']:.4f}, "
                            f"margin={metrics['reward_margin']:.4f}"
                        )
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "model_config": config},
                Path(args.checkpoint_dir) / "dpo_minigpt.pt",
            )

    model.eval().to(device)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    print(
        "Sample:",
        tokenizer.decode(
            generate(model, ids, max_new_tokens=80, temperature=0.8, top_k=40)[
                0
            ].tolist()
        ),
    )


if __name__ == "__main__":
    main()
