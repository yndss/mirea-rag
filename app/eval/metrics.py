from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


@dataclass(frozen=True)
class OverlapMetrics:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class EvalMetrics:
    bert_score: float | None
    precision: float
    recall: float
    f1: float
    rouge_1: float
    rouge_l: float


def token_set_prf1(reference: str, prediction: str) -> OverlapMetrics:
    ref = set(_tokenize(reference))
    pred = set(_tokenize(prediction))

    if not ref and not pred:
        return OverlapMetrics(precision=1.0, recall=1.0, f1=1.0)
    if not pred:
        return OverlapMetrics(precision=0.0, recall=0.0, f1=0.0)
    if not ref:
        return OverlapMetrics(precision=0.0, recall=0.0, f1=0.0)

    overlap = len(ref & pred)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(ref) if ref else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return OverlapMetrics(precision=precision, recall=recall, f1=f1)


def rouge_1_f1(reference: str, prediction: str) -> float:
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((ref_counter & pred_counter).values())

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    return (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    return (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]
