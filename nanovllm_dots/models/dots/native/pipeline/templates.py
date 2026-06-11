"""Generation templates (the prefix constants + RUNTIME_TEMPLATE_BY_NAME), copied
from the reference dots_tts/data/pipelines/tts_pipeline.py + runtime.py."""
from __future__ import annotations

TTS_TEXT_PREFIX = "[文本]"
TTS_AUDIO_PREFIX = "[文本对应语音]"
TTS_INSTRUCTION_TEXT_PREFIX = "[带指令文本]"
TTA_TEXT_PREFIX = "[声音描述]"
TTA_AUDIO_PREFIX = "[描述对应声音]"
TTS_INTERLEAVE_PREFIX = "[流式语音合成]"

DEFAULT_TRAIN_TEMPLATE = f"{TTS_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
DEFAULT_INSTRUCTION_TTS_TEMPLATE = (
    f"{TTS_INSTRUCTION_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}")
DEFAULT_TEXT_TO_AUDIO_TEMPLATE = f"{TTA_TEXT_PREFIX}{{text}}{TTA_AUDIO_PREFIX}{{audio}}"
DEFAULT_INTERLEAVE_TRAIN_TEMPLATE = f"{TTS_INTERLEAVE_PREFIX}{{interleave}}"

RUNTIME_TEMPLATE_BY_NAME = {
    "tts": DEFAULT_TRAIN_TEMPLATE,
    "instruction_tts": DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    "text_to_audio": DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    "tts_interleave": DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
}
