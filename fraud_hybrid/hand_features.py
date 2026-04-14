"""
Ручные признаки для гибридного классификатора.

Считает:
- мягкие скоринги по группам триггеров (токены/фразы/ё→е)
- простые числовые мета-признаки (длина, цифры, !/?, токены, пунктуация)
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence

import numpy as np

from fraud_hybrid.text_norm import normalize_for_match

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    d = sum(c.isdigit() for c in s)
    return d / len(s)


def _upper_ratio(s: str) -> float:
    if not s:
        return 0.0
    u = sum(c.isupper() for c in s if c.isalpha())
    a = sum(1 for c in s if c.isalpha())
    return u / a if a else 0.0


def _token_set(t_match: str) -> set[str]:
    return {x.lower().replace("ё", "е") for x in _TOKEN_RE.findall(t_match)}


def _keyword_hits(t_match: str, kw_raw: str) -> int:
    kw = kw_raw.strip().lower().replace("ё", "е")
    if not kw:
        return 0
    if " " in kw:
        return 1 if kw in t_match else 0
    if "-" in kw and not kw.startswith("-"):
        if kw in t_match:
            return 1
        parts = [p for p in kw.split("-") if p]
        if len(parts) >= 2:
            tok = _token_set(t_match)
            if all(p in tok for p in parts):
                return 1
        return 0
    tok = _token_set(t_match)
    if kw in tok:
        return 1
    if len(kw) >= 5 and kw in t_match:
        return 1
    return 0


def trigger_group_scores(text: str, groups: Sequence[Sequence[str]]) -> np.ndarray:
    t_match = normalize_for_match(text)
    out = np.zeros(len(groups), dtype=np.float32)
    for i, words in enumerate(groups):
        hits = 0
        for w in words:
            hits += _keyword_hits(t_match, w)
        if hits == 0:
            out[i] = 0.0
        elif hits == 1:
            out[i] = 0.55
        else:
            out[i] = min(1.0, 0.55 + 0.25 * (hits - 1))
    return out


def numeric_meta(text: str) -> np.ndarray:
    """Дополнительные скаляры для MLP."""
    raw = text or ""
    t_match = normalize_for_match(raw)
    tokens = _token_set(t_match)
    n = len(raw)
    log_len = math.log1p(max(n, 0))
    excl = raw.count("!")
    quest = raw.count("?")
    punct = sum(1 for c in raw if c in "@#$%^&*()[]{}|\\/<>«»\"'")
    punct_r = min(punct / max(len(raw), 1) * 8.0, 1.0)
    w_log = math.log1p(len(tokens))
    return np.array(
        [
            log_len / 10.0,
            min(_digit_ratio(raw) * 3.0, 1.0),
            _upper_ratio(raw),
            min(excl / 5.0, 1.0),
            min(quest / 5.0, 1.0),
            min(w_log / 6.0, 1.0),
            punct_r,
        ],
        dtype=np.float32,
    )


def build_hand_matrix(texts: List[str], groups: Sequence[Sequence[str]]) -> np.ndarray:
    rows = []
    for tx in texts:
        g = trigger_group_scores(tx, groups)
        m = numeric_meta(tx)
        rows.append(np.concatenate([g, m], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def hand_dim(num_groups: int) -> int:
    return num_groups + 7
