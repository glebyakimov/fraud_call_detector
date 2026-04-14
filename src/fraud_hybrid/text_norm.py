"""
Нормализация текста для TF‑IDF и поиска триггеров.
"""

from __future__ import annotations

import re


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_for_match(s: str) -> str:
    s = normalize_text(s)
    return s.replace("ё", "е")

