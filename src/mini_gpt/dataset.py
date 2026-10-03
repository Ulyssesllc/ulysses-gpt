from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

import tiktoken
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset


def clean_text(text: str) -> str:
    """Normalize Unicode and whitespace, removing nulls and control codes."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        char
        for char in text
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_hf_dataset(
    dataset: HFDataset,
    text_column: str = "text",
    min_characters: int = 1,
) -> HFDataset:
    """Clean a Hugging Face Dataset and discard empty or too-short rows."""
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Text column {text_column!r} not found. Available columns: "
            f"{dataset.column_names}"
        )
    if min_characters < 1:
        raise ValueError("min_characters must be at least 1")

    def clean_row(row: dict[str, object]) -> dict[str, str]:
        value = row[text_column]
        return {text_column: clean_text(value if isinstance(value, str) else "")}

    cleaned = dataset.map(clean_row)
    return cleaned.filter(lambda row: len(row[text_column]) >= min_characters)


class BPETokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)


class CharTokenizer:
    def __init__(self, vocabulary: str):
        self.vocabulary = vocabulary
        self.stoi = {char: index for index, char in enumerate(vocabulary)}
        self.itos = list(vocabulary)
        self.vocab_size = len(vocabulary)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[token] for token in tokens)


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
