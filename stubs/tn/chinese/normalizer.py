"""Stub for WeTextProcessing's tn.chinese.normalizer.

dots.tts imports the class at module load but only instantiates it when
normalize_text=True. Our golden/inference path uses normalize_text=False,
so this no-op stub avoids the heavy pynini/OpenFst build.
"""


class Normalizer:  # noqa: D401
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "tn.chinese.Normalizer is a stub (WeTextProcessing not installed). "
            "Run with normalize_text=False, or install WeTextProcessing for normalization."
        )

    def normalize(self, text, *args, **kwargs):
        return text
