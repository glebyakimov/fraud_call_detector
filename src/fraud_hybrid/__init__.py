"""
Пакет `fraud_hybrid`: текстовый детектор мошенничества.

Сюда попадает уже **текст**, полученный из ASR. Дальше пакет строит признаки и выдаёт метку 0/1.

Основной класс для внешнего кода:
- `HybridFraudClassifier` (fit/predict/save/load)
"""

from .hybrid_classifier import HybridFraudClassifier

__all__ = ["HybridFraudClassifier"]

