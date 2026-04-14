"""
Гибрид: [TF-IDF | ручные признаки] -> MLP на GPU.
Метки: 0 = мошенничество, 1 = не мошенничество (как в задании).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from fraud_hybrid.hand_features import build_hand_matrix, hand_dim
from fraud_hybrid.text_norm import normalize_text
from fraud_hybrid.trigger_lexicon import load_groups


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridFraudClassifier:
    def __init__(
        self,
        *,
        tfidf_max_features: int = 3500,
        tfidf_ngram_max: int = 2,
        hidden: int = 256,
        dropout: float = 0.25,
        lexicon_path: Optional[Path] = None,
    ) -> None:
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_max = tfidf_ngram_max
        self.hidden = hidden
        self.dropout = dropout
        self.lexicon_path = lexicon_path

        self.groups = load_groups(lexicon_path)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._model: Optional[_MLPHead] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._in_dim: Optional[int] = None

    def _ensure_fitted(self) -> None:
        if self._vectorizer is None or self._model is None:
            raise RuntimeError("Модель не обучена. Вызовите fit().")

    @property
    def device(self) -> torch.device:
        return self._device

    def _transform_texts(self, texts: Sequence[str]) -> np.ndarray:
        assert self._vectorizer is not None
        norm = [normalize_text(t) for t in texts]
        X_tf = self._vectorizer.transform(norm)
        if hasattr(X_tf, "toarray"):
            X_tf = X_tf.toarray()
        X_tf = np.asarray(X_tf, dtype=np.float32)
        X_h = build_hand_matrix(list(texts), self.groups)
        return np.hstack([X_tf, X_h]).astype(np.float32)

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_ratio: float = 0.15,
        seed: int = 42,
        class_weight: bool = True,
    ) -> dict[str, Any]:
        """
        labels: 0 = мошенничество, 1 = легитимный разговор.
        """
        texts = list(texts)
        y = np.asarray(labels, dtype=np.int64)
        norm = [normalize_text(t) for t in texts]

        self._vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, self.tfidf_ngram_max),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        X_tf = self._vectorizer.fit_transform(norm)
        if hasattr(X_tf, "toarray"):
            X_tf = X_tf.toarray()
        X_tf = np.asarray(X_tf, dtype=np.float32)
        X_h = build_hand_matrix(texts, self.groups)
        X = np.hstack([X_tf, X_h]).astype(np.float32)
        self._in_dim = X.shape[1]

        uniq = np.unique(y)
        n = X.shape[0]
        counts = {int(c): int((y == c).sum()) for c in uniq}
        min_cls = min(counts.values()) if counts else 0
        # Стратификация только если классов хватает на train и val
        use_strat = len(uniq) > 1 and min_cls >= 2 and n >= 12
        strat = y if use_strat else None
        if n < 8:
            X_tr, y_tr, X_va, y_va = X, y, X, y
        else:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=val_ratio, random_state=seed, stratify=strat
            )

        self._model = _MLPHead(self._in_dim, self.hidden, self.dropout).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(
            weight=self._class_weights(y_tr).to(self._device) if class_weight else None
        )

        X_tr_t = torch.from_numpy(X_tr).to(self._device)
        y_tr_t = torch.from_numpy(y_tr).to(self._device)
        n = X_tr_t.shape[0]
        history: dict[str, list[float]] = {"loss": [], "val_acc": []}

        self._model.train()
        for ep in range(epochs):
            perm = torch.randperm(n, device=self._device) if self._device.type == "cuda" else torch.randperm(n)
            total_loss = 0.0
            steps = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb = X_tr_t[idx]
                yb = y_tr_t[idx]
                opt.zero_grad()
                logits = self._model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                steps += 1
            history["loss"].append(total_loss / max(steps, 1))

            self._model.eval()
            with torch.no_grad():
                va_logits = self._model(torch.from_numpy(X_va).to(self._device))
                pred = va_logits.argmax(dim=-1).cpu().numpy()
                acc = float((pred == y_va).mean())
            history["val_acc"].append(acc)
            self._model.train()

        return {"history": history, "val_acc_final": history["val_acc"][-1] if history["val_acc"] else 0.0}

    def _class_weights(self, y: np.ndarray) -> torch.Tensor:
        # инверсия частоты класса
        n0 = max(int((y == 0).sum()), 1)
        n1 = max(int((y == 1).sum()), 1)
        w0 = len(y) / (2.0 * n0)
        w1 = len(y) / (2.0 * n1)
        return torch.tensor([w0, w1], dtype=torch.float32)

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_fitted()
        assert self._model is not None
        X = self._transform_texts(texts)
        self._model.eval()
        logits = self._model(torch.from_numpy(X).to(self._device))
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        return prob

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        proba = self.predict_proba(texts)
        return proba.argmax(axis=-1).astype(np.int64)

    def save(self, dir_path: Path) -> None:
        self._ensure_fitted()
        assert self._model is not None and self._vectorizer is not None and self._in_dim is not None
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), dir_path / "mlp.pt")
        joblib.dump(self._vectorizer, dir_path / "tfidf.joblib")
        meta = {
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_ngram_max": self.tfidf_ngram_max,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "in_dim": self._in_dim,
            "hand_dim": hand_dim(len(self.groups)),
            "num_groups": len(self.groups),
        }
        (dir_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        groups_path = dir_path / "groups.json"
        groups_path.write_text(json.dumps(self.groups, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: Path) -> HybridFraudClassifier:
        dir_path = Path(dir_path)
        meta = json.loads((dir_path / "meta.json").read_text(encoding="utf-8"))
        obj = cls(
            tfidf_max_features=meta["tfidf_max_features"],
            tfidf_ngram_max=meta["tfidf_ngram_max"],
            hidden=meta["hidden"],
            dropout=meta["dropout"],
        )
        gp = dir_path / "groups.json"
        if gp.is_file():
            obj.groups = json.loads(gp.read_text(encoding="utf-8"))
        else:
            obj.groups = load_groups(obj.lexicon_path)
        obj._vectorizer = joblib.load(dir_path / "tfidf.joblib")
        obj._in_dim = meta["in_dim"]
        obj._model = _MLPHead(obj._in_dim, obj.hidden, obj.dropout)
        sd = torch.load(dir_path / "mlp.pt", map_location=obj._device)
        obj._model.load_state_dict(sd)
        obj._model.to(obj._device)
        obj._model.eval()
        return obj
