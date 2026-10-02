import tiktoken
import torch
from torch.utils.data import Dataset


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
