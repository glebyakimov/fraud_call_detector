from __future__ import annotations

import re


def normalize_text(s: str) -> str:
    """Нижний регистр, схлопывание пробелов — для TF-IDF."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_for_match(s: str) -> str:
    """
    Текст для поиска триггеров: ё→е, нижний регистр, пробелы.
    Не трогаем цифры и латиницу внутри слов (cvv, iban).
    """
    s = normalize_text(s)
    return s.replace("ё", "е")
