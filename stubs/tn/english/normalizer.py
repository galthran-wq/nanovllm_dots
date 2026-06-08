"""Stub for WeTextProcessing's tn.english.normalizer (see chinese stub)."""


class Normalizer:  # noqa: D401
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "tn.english.Normalizer is a stub (WeTextProcessing not installed). "
            "Run with normalize_text=False, or install WeTextProcessing for normalization."
        )

    def normalize(self, text, *args, **kwargs):
        return text
