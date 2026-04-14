"""
CLI для проверки **одного** `.wav`.

Что делает этот файл:
- Берёт аудио `.wav`
- Гонит его через ASR (см. `src/asr.py`) → получает текст
- Подаёт текст в классификатор (см. `src/fraud_hybrid/hybrid_classifier.py`)
- Печатает итоговую метку:
  - `0` — мошенник
  - `1` — не мошенник

Зачем нужен отдельно от пакетного режима:
- Проверяющему обычно дают **один** файл и ждут **один** ответ.
- Поэтому максимально простой запуск: `python app\\predict_one.py file.wav`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.asr import transcribe_file
from src.fraud_hybrid.hybrid_classifier import HybridFraudClassifier

# Дефолтный “готовый” чекпоинт, который лежит в репозитории.
# Там сохранены:
# - `tfidf.joblib` (векторизатор)
# - `mlp.pt` (веса нейросети)
# - `meta.json` и `groups.json` (метаданные и список триггер-групп)
DEFAULT_CHECKPOINT = Path("checkpoints/hybrid_from_train_plus_test")


def main() -> None:
    # Аргументы CLI сделаны так, чтобы проверяющий мог:
    # - не указывать чекпоинт (берётся DEFAULT_CHECKPOINT)
    # - при желании выбрать ветку ASR-модели (revision)
    # - при отладке распечатать распознанный текст (--print-text)
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

    # 1) Загружаем классификатор (быстро: просто читаем файлы чекпоинта).
    clf = HybridFraudClassifier.load(args.checkpoint)

    # 2) Распознаём речь → текст.
    # Важно: ASR может быть “тяжёлым” по времени, а классификатор — лёгкий.
    text = transcribe_file(args.wav, revision=args.revision)
    if args.print_text:
        import sys

        print(text, file=sys.stderr)

    # 3) Классификация текста → 0/1.
    label = int(clf.predict([text])[0])
    print(label)


if __name__ == "__main__":
    main()

