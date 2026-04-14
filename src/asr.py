"""
ASR (WAV -> текст).

Загружает GigaAM-v3 с Hugging Face и распознаёт текст из WAV. Длинные записи режутся на чанки,
после чего результаты склеиваются в одну строку.

Ключевые детали реализации (важно для Windows/Transformers):
- Мы читаем/пишем WAV через `soundfile`, а не `torchaudio.load/save`, потому что на Windows
  у `torchaudio` бывают проблемы с `torchcodec`/DLL.
- Внутри `transformers` в некоторых версиях есть оптимизация с устройством `meta`
  (создание модели “без настоящих весов” для экономии RAM). Для GigaAM remote-code это ломалось,
  потому что внутри модели создаются буферы torchaudio на CPU → конфликт `cpu` vs `meta`.
  Поэтому ниже есть патч `PreTrainedModel.get_init_context`, который отключает `meta`-инициализацию
  **только** для `GigaAMModel`.
- Ещё один патч: у GigaAM remote-code иногда отсутствует `all_tied_weights_keys`, который ожидает
  `transformers` при финализации загрузки. Мы добавляем пустой атрибут перед финализацией.

Поток данных:
`wav file` → `_load_mono_16k` (моно + 16 kHz) → (если длинно) чанки по ~24 сек →
временные `.wav` → `model.transcribe(path)` → текст → склейка.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

# --- Константы препроцессинга аудио ---
_SAMPLE_RATE = 16_000
_MAX_CHUNK_SECONDS = 24
_MAX_CHUNK_SAMPLES = _MAX_CHUNK_SECONDS * _SAMPLE_RATE

DEFAULT_MODEL_ID = "ai-sage/GigaAM-v3"
DEFAULT_REVISION = "e2e_rnnt"

# Кэш модели внутри процесса: чтобы не перезагружать HF-модель на каждом файле.
_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None

# Сохраняем оригинальные методы transformers, чтобы после загрузки вернуть как было.
_orig_get_init_context: Any = None
_orig_finalize_fn: Any = None


def _gigaam_get_init_context(
    cls,
    dtype: torch.dtype,
    is_quantized: bool,
    _is_ds_init_called: bool,
    allow_all_kernels: bool | None,
):
    assert _orig_get_init_context is not None
    if cls.__name__ != "GigaAMModel":
        return _orig_get_init_context.__func__(
            cls,
            dtype,
            is_quantized,
            _is_ds_init_called,
            allow_all_kernels,
        )

    # Для GigaAMModel мы не добавляем torch.device("meta") (кроме ветки quantized),
    # чтобы избежать ошибки device mismatch при создании внутренних компонентов.
    from transformers import initialization as init
    from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled
    from transformers.integrations.hub_kernels import allow_all_hub_kernels
    from transformers.modeling_utils import (
        local_torch_dtype,
        set_quantized_state,
        set_zero3_state,
    )
    from transformers.monkey_patching import apply_patches
    from transformers.utils import logging as hf_logging

    logger = hf_logging.get_logger(__name__)
    init_contexts = [
        local_torch_dtype(dtype, cls.__name__),
        init.no_tie_weights(),
        apply_patches(),
    ]
    if allow_all_kernels:
        init_contexts.append(allow_all_hub_kernels())
    if is_deepspeed_zero3_enabled():
        import deepspeed

        if not is_quantized and not _is_ds_init_called:
            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            init_contexts.extend(
                [
                    init.no_init_weights(),
                    deepspeed.zero.Init(config_dict_or_path=deepspeed_config()),
                    set_zero3_state(),
                ]
            )
        elif is_quantized:
            init_contexts.extend([torch.device("meta"), set_quantized_state()])
    return init_contexts


def _gigaam_finalize_model_loading(model, load_config, loading_info):
    assert _orig_finalize_fn is not None
    # У некоторых версий remote-code GigaAM нет атрибута, который ожидает transformers.
    if getattr(model.config, "model_type", None) == "gigaam" and not hasattr(
        model, "all_tied_weights_keys"
    ):
        model.all_tied_weights_keys = {}
    return _orig_finalize_fn(model, load_config, loading_info)


def get_device() -> torch.device:
    global _device
    # Приоритет можно задать через переменную окружения:
    # - GIGAAM_DEVICE=cpu
    # - GIGAAM_DEVICE=cuda
    if _device is None:
        prefer_cuda = os.environ.get("GIGAAM_DEVICE", "").lower()
        if prefer_cuda == "cpu":
            _device = torch.device("cpu")
        elif prefer_cuda == "cuda" and torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    global _model, _orig_get_init_context, _orig_finalize_fn
    if _model is None:
        # Модель подгружается из Hugging Face (скачается в локальный кеш при первом запуске).
        from transformers import AutoModel
        from transformers.modeling_utils import PreTrainedModel

        dev = torch.device(device) if device is not None else get_device()
        if _orig_get_init_context is None:
            _orig_get_init_context = PreTrainedModel.get_init_context
        if _orig_finalize_fn is None:
            desc = PreTrainedModel.__dict__["_finalize_model_loading"]
            _orig_finalize_fn = (
                desc.__func__ if isinstance(desc, staticmethod) else desc
            )
        PreTrainedModel.get_init_context = classmethod(_gigaam_get_init_context)  # type: ignore[assignment]
        PreTrainedModel._finalize_model_loading = staticmethod(  # type: ignore[assignment]
            _gigaam_finalize_model_loading
        )
        try:
            _model = AutoModel.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
            )
        finally:
            # Очень важно вернуть оригинальные методы, чтобы другие модели/части кода
            # не работали “с нашим патчем”.
            PreTrainedModel.get_init_context = _orig_get_init_context
            PreTrainedModel._finalize_model_loading = staticmethod(_orig_finalize_fn)  # type: ignore[assignment]
        _model = _model.to(dev)
        _model.eval()
    return _model


def _load_mono_16k(wav_path: Path) -> torch.Tensor:
    # soundfile читает wav устойчиво на Windows, возвращает float32 [-1..1] (обычно).
    # always_2d=True → [num_samples, num_channels]
    data, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != _SAMPLE_RATE:
        # Resample делаем через torchaudio (это работает стабильно).
        wav = torchaudio.functional.resample(wav, sr, _SAMPLE_RATE)
    return wav


def _save_wav_mono_16k(path: str, wav: torch.Tensor) -> None:
    # GigaAM transcribe ожидает путь к wav-файлу, поэтому мы сохраняем временный wav на диск.
    x = wav.squeeze(0).detach().cpu().numpy()
    sf.write(path, x, _SAMPLE_RATE, subtype="PCM_16")


def _transcribe_path(model: torch.nn.Module, path: str) -> str:
    # Метод `transcribe` — это часть remote-code модели GigaAM.
    out = model.transcribe(path)
    if out is None:
        return ""
    return str(out).strip()


def transcribe_file(
    wav_path: Union[str, Path],
    *,
    model: Optional[torch.nn.Module] = None,
    revision: str = DEFAULT_REVISION,
) -> str:
    # Здесь сознательно принимаем и str и Path, чтобы CLI-скрипты были проще.
    path = Path(wav_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    # Поднимаем модель (или используем уже загруженную), затем работаем только с аудио-тензором.
    m = model or load_model(revision=revision)
    wav = _load_mono_16k(path)
    n = wav.shape[-1]

    if n <= _MAX_CHUNK_SAMPLES:
        # Короткий файл: просто транскрибируем одним куском.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _save_wav_mono_16k(tmp_path, wav.cpu())
            return _transcribe_path(m, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Длинный файл: режем на чанки фиксированного размера.
    parts: list[str] = []
    start = 0
    while start < n:
        end = min(start + _MAX_CHUNK_SAMPLES, n)
        chunk = wav[:, start:end]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _save_wav_mono_16k(tmp_path, chunk.cpu())
            text = _transcribe_path(m, tmp_path)
            if text:
                parts.append(text)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        start = end

    return " ".join(parts).strip()

