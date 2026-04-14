"""
CLI: один WAV -> один label.

Печатает только 0/1 (0 = мошенник, 1 = не мошенник). Опционально может вывести распознанный текст в stderr.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.asr import transcribe_file
from src.fraud_hybrid.hybrid_classifier import HybridFraudClassifier

DEFAULT_CHECKPOINT = Path("checkpoints/hybrid_from_train_plus_test")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict: one wav -> label (0 fraud, 1 not fraud)")
    ap.add_argument("wav", type=Path, help="Путь к .wav")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Папка чекпоинта (по умолчанию {DEFAULT_CHECKPOINT.as_posix()})",
    )
    ap.add_argument("--revision", default="e2e_rnnt", help="Ветка GigaAM на HF")
    ap.add_argument("--print-text", action="store_true", help="Печатать распознанный текст в stderr")
    args = ap.parse_args()

    clf = HybridFraudClassifier.load(args.checkpoint)
    text = transcribe_file(args.wav, revision=args.revision)
    if args.print_text:
        import sys

        print(text, file=sys.stderr)
    label = int(clf.predict([text])[0])
    print(label)


if __name__ == "__main__":
    main()

