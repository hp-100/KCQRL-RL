"""Metrics for paired CAT benchmark evaluation."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence


def auc_score(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    positives = sum(1 for y in y_true if y >= 0.5)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranked = sorted(zip(y_score, y_true), key=lambda p: p[0])
    pos_rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_rank_sum += avg_rank * sum(1 for _, y in ranked[i:j] if y >= 0.5)
        i = j
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def nll_score(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    if not y_true:
        return float("nan")
    eps = 1e-7
    total = 0.0
    for y, p in zip(y_true, y_score):
        p = min(max(float(p), eps), 1.0 - eps)
        y = float(y >= 0.5)
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(y_true)


def accuracy_score(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    if not y_true:
        return float("nan")
    return sum(int((p >= 0.5) == (y >= 0.5)) for y, p in zip(y_true, y_score)) / len(y_true)


def brier_score(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    if not y_true:
        return float("nan")
    return sum((min(max(float(p), 1e-7), 1 - 1e-7) - float(y >= 0.5)) ** 2 for y, p in zip(y_true, y_score)) / len(y_true)


def gini(values: Iterable[int]) -> float:
    xs = sorted(float(v) for v in values if v >= 0)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * weighted) / (n * sum(xs)) - (n + 1) / n


def metric_bundle(y_true: Sequence[float], y_score: Sequence[float]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_score),
        "auc": auc_score(y_true, y_score),
        "nll": nll_score(y_true, y_score),
        "brier": brier_score(y_true, y_score),
    }


def nanmean(vals: Sequence[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")
