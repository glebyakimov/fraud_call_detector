"""
Нормализация текста для TF‑IDF и поиска триггеров.

Здесь две функции, потому что у нас две разные задачи:
- `normalize_text` — “мягкая” нормализация для TF‑IDF:
  - lower()
  - схлопывание пробелов
- `normalize_for_match` — нормализация для поиска триггеров:
  - делает всё как `normalize_text`
  - дополнительно приводит `ё` → `е`, чтобы “счёт/счет” матчились одинаково
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

