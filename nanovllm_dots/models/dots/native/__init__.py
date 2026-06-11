"""Native reimplementation of the dots.tts model components.

These modules replace the reference ``dots_tts`` neural nets at runtime so the
inference engine / server no longer imports the reference package. They mirror the
reference module structure and attribute names exactly (UNFUSED q/k/v, same leaf
names) so the engine's accelerators (flash_cached_fm, batched_fm, flash_patch_
encoder, ...) keep binding to the same leaves, and weights load with a near-direct
state-dict copy.

Each component is validated native-vs-reference at cos ~1.0 in tests/test_native_*.
The reference stays only as a dev/test dependency.
"""
