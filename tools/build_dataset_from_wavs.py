"""
Утилита (не обязательна для сдачи): WAV (Fraud/NotFraud) -> CSV (filename;label;text) для обучения.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.asr import load_model, transcribe_file


def _iter_wavs(dir_path: Path) -> list[Path]:
    wavs = list(dir_path.glob("*.wav")) + list(dir_path.glob("*.WAV"))
    uniq: dict[str, Path] = {}
    for p in wavs:
        uniq[str(p.resolve()).lower()] = p
    return sorted(uniq.values())


def _transcribe_safe(wav: Path, revision: str) -> str:
    try:
        return transcribe_file(wav, revision=revision)
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="WAV -> (ASR text, label) -> CSV для обучения")
    ap.add_argument("--root", type=Path, required=True, help="Корень с папками классов")
    ap.add_argument("--fraud-dir", default="Fraud", help="Имя папки с мошенническими WAV (label=0)")
    ap.add_argument("--not-fraud-dir", default="NotFraud", help="Имя папки с не мошенническими WAV (label=1)")
    ap.add_argument("--out", type=Path, required=True, help="Выходной CSV (delimiter=';')")
    ap.add_argument("--revision", default="e2e_rnnt", help="Ветка GigaAM на HF")
    ap.add_argument("--device", default=None, help="cpu|cuda (иначе авто)")
    ap.add_argument("--limit", type=int, default=0, help="Ограничить число файлов на класс (0 = без лимита)")
    args = ap.parse_args()

    root: Path = args.root
    fraud_dir = root / args.fraud_dir
    not_fraud_dir = root / args.not_fraud_dir
    if not fraud_dir.is_dir():
        raise SystemExit(f"Нет папки: {fraud_dir}")
    if not not_fraud_dir.is_dir():
        raise SystemExit(f"Нет папки: {not_fraud_dir}")

    other_dirs = sorted(
        d.name for d in root.iterdir() if d.is_dir() and d.name not in {args.fraud_dir, args.not_fraud_dir}
    )
    if other_dirs:
        print(f"Пропускаю прочие подпапки: {', '.join(other_dirs)}")

    load_model(revision=args.revision, device=args.device)

    fraud_wavs = _iter_wavs(fraud_dir)
    not_wavs = _iter_wavs(not_fraud_dir)
    if args.limit and args.limit > 0:
        fraud_wavs = fraud_wavs[: args.limit]
        not_wavs = not_wavs[: args.limit]

    rows: list[tuple[str, int, str]] = []
    total = len(fraud_wavs) + len(not_wavs)
    done = 0

    for wav in fraud_wavs:
        done += 1
        print(f"[{done}/{total}] ASR fraud: {wav.name}")
        rows.append((wav.name, 0, _transcribe_safe(wav, revision=args.revision)))

    for wav in not_wavs:
        done += 1
        print(f"[{done}/{total}] ASR not_fraud: {wav.name}")
        rows.append((wav.name, 1, _transcribe_safe(wav, revision=args.revision)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["filename", "label", "text"])
        w.writerows(rows)

    print(f"Готово: {len(rows)} строк -> {args.out.resolve()}")


if __name__ == "__main__":
    main()

