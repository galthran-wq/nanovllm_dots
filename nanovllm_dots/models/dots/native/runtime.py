"""Native DotsRuntime: the input pipeline the engine calls (_load_prompt_audio,
_prepare_inputs) + from_pretrained that assembles the whole native model. Ported
from the reference dots_tts/runtime.py (the methods the engine actually uses).
"""
from __future__ import annotations

import hashlib

import librosa
import torch

from .config import load_config
from .core import DotsCore
from .loader import build_native_speaker, build_native_vocoder
from .model import DotsModel
from .pipeline.templates import RUNTIME_TEMPLATE_BY_NAME
from .pipeline.text import (attach_language_tag, detect, normalize_language_code,
                            normalize_text)
from .pipeline.tokenizing import build_generation_schedule
from .speaker.audio import high_quality_resample


class DotsRuntime:
    def __init__(self, model: DotsModel, *, device="cuda", max_generate_length: int = 256):
        self.model = model
        self.device = torch.device(device)
        self.max_generate_length = int(max_generate_length)
        self.sample_rate = int(model.vocoder.sample_rate)

    @classmethod
    def from_pretrained(cls, model_dir: str, *, device="cuda", dtype=torch.bfloat16,
                        max_generate_length: int = 256) -> "DotsRuntime":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        core = DotsCore.from_pretrained(model_dir, tokenizer, device=device, dtype=dtype)
        vocoder = build_native_vocoder(model_dir, device=device)
        speaker = build_native_speaker(model_dir, device=device)
        cfg = load_config(model_dir)
        model = DotsModel(core, vocoder, speaker, patch_size=cfg["patch_size"])
        return cls(model, device=device, max_generate_length=max_generate_length)

    # ----------------------------------------------------------- prompt audio
    def _load_prompt_audio(self, prompt_audio_path: str) -> torch.Tensor:
        prompt_audio, sample_rate = librosa.load(prompt_audio_path, sr=None, mono=True)
        prompt_audio = librosa.effects.trim(prompt_audio, top_db=30)[0]
        prompt_audio = torch.from_numpy(prompt_audio).unsqueeze(0)
        prompt_audio = high_quality_resample(prompt_audio, orig_sr=sample_rate,
                                             target_sr=self.sample_rate)
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        return prompt_audio

    # ----------------------------------------------------------- text / language
    def _resolve_language(self, language: str | None, *, text: str) -> str | None:
        if language is None:
            return None
        stripped = language.strip()
        if not stripped or stripped.lower() == "none":
            return None
        if stripped.lower() == "auto_detect":
            return normalize_language_code(detect(text))
        normalized_language = normalize_language_code(stripped)
        if normalized_language is None:
            raise ValueError(
                f"Unsupported language={language!r}. Expected 'none', 'auto_detect', "
                "or a valid language code/name.")
        return normalized_language

    def _process_prompt_text(self, prompt_text: str | None, *, language: str | None = None) -> str:
        if prompt_text is None:
            return ""
        prompt_text = prompt_text.strip()
        if not prompt_text:
            return ""
        prompt_language = language
        if prompt_language is None:
            prompt_language = normalize_language_code(detect(prompt_text))
        if prompt_language not in {"ZH", "YUE", "JA", "口音:粤语"}:
            prompt_text += " "
        if language is not None:
            prompt_text = attach_language_tag(prompt_text, language)
        return prompt_text

    def _process_text(self, text: str, *, language: str | None = None,
                      normalize: bool = False) -> tuple[str, str | None]:
        stripped = text.strip()
        if normalize:
            stripped = normalize_text(stripped)
        resolved_language = self._resolve_language(language, text=stripped)
        return stripped, resolved_language

    def _estimate_prompt_audio_patch_count(self, *, prompt_audio, prompt_text: str) -> int:
        if prompt_audio is None or not prompt_text:
            return 0
        samples_per_patch = int(self.model.config.patch_size * self.model.hop_size)
        prompt_samples = int(prompt_audio.shape[-1])
        return (prompt_samples + samples_per_patch - 1) // samples_per_patch

    def _normalize_template_name(self, template_name: str | None) -> str:
        if template_name is None:
            return "tts"
        if template_name not in RUNTIME_TEMPLATE_BY_NAME:
            raise ValueError(f"Unknown template_name={template_name!r}. "
                             f"Expected one of {sorted(RUNTIME_TEMPLATE_BY_NAME)}.")
        return template_name

    @staticmethod
    def _build_request_id(**parts) -> str:
        h = hashlib.sha1(repr(sorted(parts.items())).encode("utf-8")).hexdigest()
        return h[:16]

    # ----------------------------------------------------------- the entry point
    def _prepare_inputs(self, *, text: str, prompt_audio_path: str | None,
                        prompt_text: str | None, template_name: str | None,
                        language: str | None = None, normalize_text: bool = False) -> dict:
        normalized_template_name = self._normalize_template_name(template_name)
        template = RUNTIME_TEMPLATE_BY_NAME[normalized_template_name]
        if prompt_text and not prompt_audio_path:
            raise ValueError("prompt_text requires prompt_audio_path.")

        normalized_text, normalized_language = self._process_text(
            text, language=language, normalize=normalize_text)
        normalized_prompt_text = self._process_prompt_text(prompt_text, language=normalized_language)
        if normalized_language is not None and not normalized_prompt_text:
            normalized_text = attach_language_tag(normalized_text, normalized_language)

        inputs: dict = {
            "fid": self._build_request_id(
                text=normalized_text, prompt_audio_path=prompt_audio_path,
                prompt_text=normalized_prompt_text, template_name=normalized_template_name,
                language=normalized_language),
            "language": normalized_language or "",
            "text": normalized_text,
            "prompt_text": normalized_prompt_text,
            "template_name": normalized_template_name,
        }

        if prompt_audio_path:
            inputs["prompt_audio"] = self._load_prompt_audio(prompt_audio_path)
        prompt_audio_patch_count = self._estimate_prompt_audio_patch_count(
            prompt_audio=inputs.get("prompt_audio"), prompt_text=normalized_prompt_text)
        if prompt_audio_patch_count > 0 and self.max_generate_length <= prompt_audio_patch_count:
            raise ValueError(
                "max_generate_length must exceed prompt audio patch count when prompt_text is "
                f"provided: max_generate_length={self.max_generate_length} "
                f"prompt_audio_patch_count={prompt_audio_patch_count}.")

        schedule_spec = build_generation_schedule(
            text=f"{normalized_prompt_text}{normalized_text}", tokenizer=self.model.core.tokenizer,
            template=template, max_audio_tokens=self.max_generate_length)
        schedule = torch.tensor(schedule_spec["schedule_ids"], dtype=torch.long, device=self.device)
        inputs["generation_schedule"] = schedule.unsqueeze(0)
        return inputs
