import argparse
import sys
import urllib.request
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mini_gpt.dataset import BPETokenizer, TextDataset, clean_text
from mini_gpt.generate import generate, generate_with_metrics
from mini_gpt.metrics import append_metrics, exact_match, rouge_l
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
    token_chunks: list[torch.Tensor] = []
    token_count = 0
    print(f" Encoding {label} (limit: {max_tokens:,} tokens)...", flush=True)
    iterator = iter(texts)
    interactive_progress = sys.stderr.isatty()
    with tqdm(
        desc=f"Tokenizing {label}",
        unit="batch",
        disable=not interactive_progress,
    ) as progress:
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
                eot_token_id = tokenizer.encoder.eot_token
                if ids[-1] != eot_token_id:
                    if len(ids) == remaining:
                        ids = ids[:-1]
                    ids.append(eot_token_id)
                if ids:
                    token_chunks.append(torch.tensor(ids, dtype=torch.int32))
                    token_count += len(ids)
                if token_count >= max_tokens:
                    break
            progress.update(1)
            if not interactive_progress and progress.n % 10 == 0:
                print(
                    f"Tokenizing {label}: batches={progress.n}, "
                    f"tokens={token_count:,}/{max_tokens:,}",
                    flush=True,
                )
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
        description="Train ulysses-gpt on configurable datasets."
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
    parser.add_argument("--context-column", default="context")
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
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=16,
        help="Maximum SFT validation responses to generate for ROUGE-L/EM.",
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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if args.checkpoint or args.resume:
        model.load_state_dict(saved.get("model_state_dict", saved))

    if args.task == "pretrain":
        if args.hf_dataset:
            print(
                f"Loading Hugging Face dataset {args.hf_dataset!r} "
                f"(split={args.hf_split!r})...",
                flush=True,
            )
            train_hf = _load_hf_split(args.hf_dataset, args.hf_config, args.hf_split)
            print(f"Loaded {len(train_hf):,} training rows.", flush=True)
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
            print(f"Using text column {text_column!r}", flush=True)
            train_hf = train_hf.shuffle(seed=args.seed)
            print("Prepared shuffled training split.", flush=True)
            train_texts = _texts(train_hf, text_column, args.min_text_characters)
            try:
                print(
                    f"Loading validation split {args.validation_split!r}...",
                    flush=True,
                )
                valid_hf = _load_hf_split(
                    args.hf_dataset, args.hf_config, args.validation_split
                )
                print(f"Loaded {len(valid_hf):,} validation rows.", flush=True)
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
                print(
                    f"Using dataset validation split {args.validation_split!r}",
                    flush=True,
                )
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
            metrics = trainer.evaluate_metrics()
            print(
                f"Epoch {epoch}: train={train_loss:.4f}, "
                f"val={metrics['loss']:.4f}, "
                f"ppl={metrics['perplexity']:.2f}, "
                f"next-token-acc={metrics['accuracy']:.2%}"
            )
            append_metrics(
                str(Path(args.checkpoint_dir) / "metrics.jsonl"),
                {
                    "stage": "pretrain",
                    "epoch": epoch,
                    "step": trainer.global_step,
                    "train_loss": train_loss,
                    "validation_cross_entropy": metrics["loss"],
                    "perplexity": metrics["perplexity"],
                    "next_token_accuracy": metrics["accuracy"],
                },
            )
            if trainer.global_step >= total_steps:
                break
        trainer.save_checkpoint("ulysses-gpt-pretrain.pt")

    else:
        print(
            f"Loading Hugging Face dataset {args.hf_dataset!r} "
            f"(split={args.hf_split!r})...",
            flush=True,
        )
        dataset = _load_hf_split(args.hf_dataset, args.hf_config, args.hf_split)
        print(f"Loaded {len(dataset):,} rows.", flush=True)
        if args.task == "sft":
            required = (args.prompt_column, args.response_column)
            missing = [col for col in required if col not in dataset.column_names]
            if missing:
                raise ValueError(
                    f"SFT columns missing: {missing}; available: {dataset.column_names}"
                )
            dataset = dataset.shuffle(seed=args.seed)
            count = min(len(dataset), args.max_samples)
            rows = [
                dataset[i]
                for i in tqdm(range(count), desc="Reading SFT rows", unit="sample")
            ]
            validation_count = max(1, int(count * 0.05)) if count > 1 else 0
            train_rows = rows[:-validation_count] if validation_count else rows
            validation_rows = rows[-validation_count:] if validation_count else []
            prompts = [_stringify(row[args.prompt_column]) for row in train_rows]
            if args.context_column in dataset.column_names:
                prompts = [
                    f"Context:\n{_stringify(row[args.context_column])}"
                    f"\n\nInstruction:\n{prompt}"
                    if _stringify(row[args.context_column]).strip()
                    else prompt
                    for prompt, row in zip(prompts, train_rows)
                ]
            responses = [_stringify(row[args.response_column]) for row in train_rows]
            validation_prompts = [
                _stringify(row[args.prompt_column]) for row in validation_rows
            ]
            if args.context_column in dataset.column_names:
                validation_prompts = [
                    f"Context:\n{_stringify(row[args.context_column])}"
                    f"\n\nInstruction:\n{prompt}"
                    if _stringify(row[args.context_column]).strip()
                    else prompt
                    for prompt, row in zip(validation_prompts, validation_rows)
                ]
            validation_responses = [
                _stringify(row[args.response_column]) for row in validation_rows
            ]
            eot_token_id = tokenizer.encoder.eot_token
            post_data = SFTDataset(
                prompts,
                responses,
                tokenizer,
                args.block_size,
                eos_token_id=eot_token_id,
            )
            loader = DataLoader(post_data, batch_size=args.batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            model.to(device).train()
            for epoch in range(args.epochs):
                total_loss = 0.0
                total_response_tokens = 0
                interactive_progress = sys.stderr.isatty()
                for batch_index, (x, y) in enumerate(
                    tqdm(
                        loader,
                        desc=f"SFT epoch {epoch + 1}",
                        disable=not interactive_progress,
                    ),
                    start=1,
                ):
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = model(x, y)
                    labels = y[:, 1:].contiguous().view(-1)
                    loss = F.cross_entropy(
                        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                        labels,
                        ignore_index=-100,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    response_tokens = int((labels != -100).sum().item())
                    total_loss += loss.item() * response_tokens
                    total_response_tokens += response_tokens
                    if not interactive_progress and batch_index % 100 == 0:
                        print(
                            f"SFT epoch {epoch + 1}: "
                            f"batch={batch_index}/{len(loader)}, "
                            f"response_loss={loss.item():.4f}",
                            flush=True,
                        )
                metrics = {
                    "stage": "sft",
                    "epoch": epoch + 1,
                    "response_loss": total_loss / max(1, total_response_tokens),
                    "response_tokens": total_response_tokens,
                }
                if validation_rows:
                    model.eval()
                    val_loss_sum = 0.0
                    val_token_count = 0
                    for start in range(0, len(validation_prompts), args.batch_size):
                        end = start + args.batch_size
                        val_batch = SFTDataset(
                            validation_prompts[start:end],
                            validation_responses[start:end],
                            tokenizer,
                            args.block_size,
                            eos_token_id=eot_token_id,
                        )
                        val_loader = DataLoader(val_batch, batch_size=args.batch_size)
                        for val_x, val_y in val_loader:
                            val_x, val_y = val_x.to(device), val_y.to(device)
                            val_logits, _ = model(val_x, val_y)
                            val_labels = val_y[:, 1:].contiguous().view(-1)
                            val_sum = F.cross_entropy(
                                val_logits[:, :-1, :]
                                .contiguous()
                                .view(-1, val_logits.size(-1)),
                                val_labels,
                                ignore_index=-100,
                                reduction="sum",
                            )
                            val_loss_sum += val_sum.item()
                            val_token_count += int((val_labels != -100).sum().item())
                    metrics["validation_response_loss"] = val_loss_sum / max(
                        1, val_token_count
                    )
                    rouge_scores = []
                    exact_matches = []
                    eval_count = min(args.eval_samples, len(validation_prompts))
                    for prompt, reference in zip(
                        validation_prompts[:eval_count],
                        validation_responses[:eval_count],
                    ):
                        prompt_ids = tokenizer.encode(prompt)[-(args.block_size - 1) :]
                        if not prompt_ids:
                            prompt_ids = [0]
                        input_ids = torch.tensor(
                            [prompt_ids], dtype=torch.long, device=device
                        )
                        output = generate(
                            model,
                            input_ids,
                            max_new_tokens=min(80, args.block_size - len(prompt_ids)),
                            temperature=0.8,
                            top_k=40,
                            eos_token_id=eot_token_id,
                        )
                        prediction = tokenizer.decode(
                            output[0, len(prompt_ids) :].tolist()
                        )
                        rouge_scores.append(rouge_l(prediction, reference))
                        exact_matches.append(exact_match(prediction, reference))
                    metrics["rouge_l"] = sum(rouge_scores) / max(1, len(rouge_scores))
                    metrics["exact_match"] = sum(exact_matches) / max(
                        1, len(exact_matches)
                    )
                    model.train()
                print(
                    f"SFT epoch {epoch + 1}: "
                    f"response_loss={metrics['response_loss']:.4f}"
                    + (
                        f", val_response_loss="
                        f"{metrics['validation_response_loss']:.4f}, "
                        f"rouge_l={metrics['rouge_l']:.4f}, "
                        f"em={metrics['exact_match']:.2%}"
                        if validation_rows
                        else ""
                    )
                )
                append_metrics(
                    str(Path(args.checkpoint_dir) / "metrics.jsonl"), metrics
                )
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "model_config": config},
                Path(args.checkpoint_dir) / "ulysses-gpt-sft.pt",
            )
        else:
            required = (args.prompt_column, args.chosen_column, args.rejected_column)
            missing = [col for col in required if col not in dataset.column_names]
            if missing:
                raise ValueError(
                    f"DPO columns missing: {missing}; available: {dataset.column_names}"
                )
            count = min(len(dataset), args.max_samples)
            rows = [
                dataset[i]
                for i in tqdm(range(count), desc="Reading DPO rows", unit="sample")
            ]
            prompts = [_stringify(row[args.prompt_column]) for row in rows]
            chosen = [_stringify(row[args.chosen_column]) for row in rows]
            rejected = [_stringify(row[args.rejected_column]) for row in rows]
            post_data = DPODataset(
                prompts,
                chosen,
                rejected,
                tokenizer,
                args.block_size,
                eos_token_id=tokenizer.encoder.eot_token,
            )
            loader = DataLoader(post_data, batch_size=args.batch_size, shuffle=True)
            trainer = DPOTrainer(model, lr=args.learning_rate, device=device)
            for epoch in range(args.epochs):
                epoch_metrics: list[dict[str, float]] = []
                for step, batch in enumerate(
                    tqdm(
                        loader,
                        desc=f"DPO epoch {epoch + 1}",
                        disable=not sys.stderr.isatty(),
                    ),
                    1,
                ):
                    metrics = trainer.train_step(batch)
                    epoch_metrics.append(metrics)
                    if step % 20 == 0:
                        print(
                            f"DPO step {step}: loss={metrics['dpo_loss']:.4f}, "
                            f"margin={metrics['reward_margin']:.4f}, "
                            f"win_rate={metrics['win_rate']:.2%}"
                        )
                summary = {
                    "stage": "dpo",
                    "epoch": epoch + 1,
                    **{
                        name: sum(item[name] for item in epoch_metrics)
                        / max(1, len(epoch_metrics))
                        for name in (
                            "dpo_loss",
                            "chosen_reward",
                            "rejected_reward",
                            "reward_margin",
                            "win_rate",
                        )
                    },
                }
                print(
                    f"DPO epoch {epoch + 1}: loss={summary['dpo_loss']:.4f}, "
                    f"margin={summary['reward_margin']:.4f}, "
                    f"win_rate={summary['win_rate']:.2%}"
                )
                append_metrics(
                    str(Path(args.checkpoint_dir) / "metrics.jsonl"), summary
                )
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "model_config": config},
                Path(args.checkpoint_dir) / "ulysses-gpt-dpo.pt",
            )

    model.eval().to(device)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    sampled, inference_metrics = generate_with_metrics(
        model,
        ids,
        max_new_tokens=80,
        temperature=0.8,
        top_k=40,
        eos_token_id=tokenizer.encoder.eot_token,
    )
    print("Sample:", tokenizer.decode(sampled[0].tolist()))
    print(
        f"Inference: TTFT={inference_metrics['ttft_seconds']:.3f}s, "
        "throughput="
        f"{inference_metrics['throughput_tokens_per_second']:.2f} tokens/s"
    )
    append_metrics(
        str(Path(args.checkpoint_dir) / "metrics.jsonl"),
        {"stage": "inference", **inference_metrics},
    )


if __name__ == "__main__":
    main()
