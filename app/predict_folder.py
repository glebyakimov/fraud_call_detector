"""
CLI для проверки **папки** с `.wav` файлами и сохранения результата в CSV.

Что делает этот файл:
- Собирает список WAV в папке
- Для каждого WAV делает ASR (см. `src/asr.py`) → получает текст
- Классифицирует текст моделью (см. `src/fraud_hybrid/hybrid_classifier.py`)
- Пишет CSV строго в формате задания:
  `Название файла;label`

Зачем это нужно:
- Удобно быстро прогнать “пачку” файлов (например, test-набор) и получить таблицу ответов.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.asr import transcribe_file
from src.fraud_hybrid.hybrid_classifier import HybridFraudClassifier

# Дефолтный чекпоинт с уже обученной моделью.
DEFAULT_CHECKPOINT = Path("checkpoints/hybrid_from_train_plus_test")


def _iter_wavs(dir_path: Path) -> list[Path]:
    # Windows часто case-insensitive, поэтому glob("*.wav") и glob("*.WAV")
    # могут пересечься. Делаем дедупликацию по нормализованному абсолютному пути.
    wavs = list(dir_path.glob("*.wav")) + list(dir_path.glob("*.WAV"))
    uniq: dict[str, Path] = {}
    for p in wavs:
        uniq[str(p.resolve()).lower()] = p
    return sorted(uniq.values())


def main() -> None:
    # CLI сделан так, чтобы:
    # - можно было указать только папку + --out
    # - при желании переопределить чекпоинт
    # - при отладке распечатать ASR-текст каждого файла в stderr
    ap = argparse.ArgumentParser(description="Predict: folder wav -> CSV")
    ap.add_argument("wav_dir", type=Path, help="Папка с .wav")
    ap.add_argument("--out", type=Path, required=True, help="Выходной CSV")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Папка чекпоинта")
    ap.add_argument("--revision", default="e2e_rnnt", help="Ветка GigaAM на HF")
    ap.add_argument("--print-text", action="store_true", help="Печатать распознанный текст каждого файла в stderr")
    args = ap.parse_args()

    wavs = _iter_wavs(args.wav_dir)
    if not wavs:
        raise SystemExit(f"Нет .wav в {args.wav_dir}")

    # Загружаем модель один раз, дальше используем для всех файлов.
    clf = HybridFraudClassifier.load(args.checkpoint)

    rows: list[tuple[str, int]] = []
    for wav in wavs:
        # 1) WAV -> текст
        text = transcribe_file(wav, revision=args.revision)
        if args.print_text:
            import sys

            print(f"[{wav.name}] {text}", file=sys.stderr)
        # 2) текст -> 0/1
        label = int(clf.predict([text])[0])
        rows.append((wav.name, label))

    # Пишем CSV с нужным разделителем `;` и заголовком.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Название файла", "label"])
        w.writerows(rows)

    print(args.out.resolve())


if __name__ == "__main__":
    main()

