"""
Утилита (не обязательна для сдачи): обучение гибрида по CSV.

См. основной код в корне проекта. Этот файл оставлен для удобства повторного обучения.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.fraud_hybrid.hybrid_classifier import HybridFraudClassifier


def read_labeled_csv(path: Path, text_col: str, label_col: str, delimiter: str) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Пустой CSV или нет заголовка")
        for row in reader:
            t = (row.get(text_col) or "").strip()
            if not t:
                continue
            lab = int(row[label_col])
            texts.append(t)
            labels.append(lab)
    if not texts:
        raise ValueError("Нет ни одной строки с текстом")
    return texts, labels


def main() -> None:
    p = argparse.ArgumentParser(description="Обучение гибрида TF-IDF + триггеры + MLP")
    p.add_argument("--csv", type=Path, required=True, help="CSV с колонками текста и метки")
    p.add_argument("--out", type=Path, required=True, help="Папка для чекпоинта")
    p.add_argument("--text-col", default="text", help="Имя колонки с текстом")
    p.add_argument("--label-col", default="label", help="Имя колонки с меткой 0/1")
    p.add_argument("--delimiter", default=";", help="Разделитель CSV")
    p.add_argument("--lexicon", type=Path, default=None, help="Файл групп триггеров (группа = строка, слова через |)")
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tfidf-dim", type=int, default=3500)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--no-class-weight", action="store_true")
    args = p.parse_args()

    texts, labels = read_labeled_csv(args.csv, args.text_col, args.label_col, args.delimiter)
    clf = HybridFraudClassifier(
        tfidf_max_features=args.tfidf_dim,
        hidden=args.hidden,
        lexicon_path=args.lexicon,
    )
    info = clf.fit(
        texts,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        class_weight=not args.no_class_weight,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    clf.save(args.out)
    print(f"Сохранено в {args.out.resolve()}")
    print(f"Финальная val accuracy (доля верных на отложенной доле): {info['val_acc_final']:.4f}")


if __name__ == "__main__":
    main()

