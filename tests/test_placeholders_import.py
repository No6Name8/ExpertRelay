"""expertrelay.cache and expertrelay.predictor are documented placeholders
(see docs/limitations.md) -- this just guards against the package being
broken (e.g. a typo making it uninstallable), not against them being empty."""

from __future__ import annotations

import importlib


def test_cache_and_predictor_import_cleanly():
    importlib.import_module("expertrelay.cache")
    importlib.import_module("expertrelay.predictor")
