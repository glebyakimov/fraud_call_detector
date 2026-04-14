"""
ASR (WAV -> текст).

Загружает GigaAM-v3 с Hugging Face и распознаёт текст из WAV. Длинные записи режутся на чанки,
после чего результаты склеиваются в одну строку.
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

_SAMPLE_RATE = 16_000
_MAX_CHUNK_SECONDS = 24
_MAX_CHUNK_SAMPLES = _MAX_CHUNK_SECONDS * _SAMPLE_RATE

DEFAULT_MODEL_ID = "ai-sage/GigaAM-v3"
DEFAULT_REVISION = "e2e_rnnt"

_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None

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
    if getattr(model.config, "model_type", None) == "gigaam" and not hasattr(
        model, "all_tied_weights_keys"
    ):
        model.all_tied_weights_keys = {}
    return _orig_finalize_fn(model, load_config, loading_info)


def get_device() -> torch.device:
    global _device
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
            PreTrainedModel.get_init_context = _orig_get_init_context
            PreTrainedModel._finalize_model_loading = staticmethod(_orig_finalize_fn)  # type: ignore[assignment]
        _model = _model.to(dev)
        _model.eval()
    return _model


def _load_mono_16k(wav_path: Path) -> torch.Tensor:
    data, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != _SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, _SAMPLE_RATE)
    return wav


def _save_wav_mono_16k(path: str, wav: torch.Tensor) -> None:
    x = wav.squeeze(0).detach().cpu().numpy()
    sf.write(path, x, _SAMPLE_RATE, subtype="PCM_16")


def _transcribe_path(model: torch.nn.Module, path: str) -> str:
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
    path = Path(wav_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    m = model or load_model(revision=revision)
    wav = _load_mono_16k(path)
    n = wav.shape[-1]

    if n <= _MAX_CHUNK_SAMPLES:
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

