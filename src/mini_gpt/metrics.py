"""Small dependency-free evaluation and metric logging helpers."""

import json
import os
from typing import Any


def append_metrics(path: str, metrics: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def normalize_answer(text: str) -> str:
    return " ".join(text.casefold().split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def rouge_l(prediction: str, reference: str) -> float:
    """Whitespace-token ROUGE-L F1; intended for lightweight local evaluation."""
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(not pred and not ref)
    previous = [0] * (len(ref) + 1)
    for token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, 1):
            current.append(
                previous[index - 1] + 1
                if token == ref_token
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)
