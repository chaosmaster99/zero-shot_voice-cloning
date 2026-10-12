from __future__ import annotations

import pytest

from backend.prosody.presets import CUSTOM_KEY, LIMITS, NEUTRAL_PARAMS, PRESETS, list_presets, resolve_prosody


def test_expected_presets_exist():
    keys = set(PRESETS)
    assert {"neutral", "happy", "sad", "angry", "calm", "excited", CUSTOM_KEY} <= keys
    assert len(list_presets()) == len(PRESETS)


def test_all_presets_are_within_limits():
    for key, spec in PRESETS.items():
        for name, value in spec["params"].items():
            low, high = LIMITS[name]
            assert low <= value <= high, f"{key}.{name}"


def test_emotional_presets_move_in_expected_directions():
    happy, sad = PRESETS["happy"]["params"], PRESETS["sad"]["params"]
    assert happy["pitch"] > 0 > sad["pitch"]
    assert happy["rate"] > 1.0 > sad["rate"]
    assert happy["energy"] > 0 > sad["energy"]
    assert happy["pause"] < 1.0 < sad["pause"]
    assert happy["pitch_range"] > 1.0 > sad["pitch_range"]


def test_neutral_is_identity():
    key, params = resolve_prosody("neutral")
    assert key == "neutral"
    assert params == NEUTRAL_PARAMS


def test_default_is_neutral():
    assert resolve_prosody(None)[0] == "neutral"


def test_preset_values_can_be_overridden():
    key, params = resolve_prosody("happy", pitch=3.0)
    assert key == "happy"
    assert params["pitch"] == 3.0
    assert params["rate"] == PRESETS["happy"]["params"]["rate"]


def test_custom_uses_supplied_values_over_neutral():
    key, params = resolve_prosody("custom", pitch=-12, rate=0.9)
    assert key == "custom"
    assert params["pitch"] == -12
    assert params["rate"] == 0.9
    assert params["energy"] == 0.0


def test_case_insensitive_preset():
    assert resolve_prosody("SAD")[0] == "sad"


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        resolve_prosody("furious")


@pytest.mark.parametrize("field,value", [("pitch", 500), ("rate", 5), ("pause", -1), ("energy", -200), ("pitch_range", 0)])
def test_out_of_range_values_rejected(field, value):
    with pytest.raises(ValueError):
        resolve_prosody("custom", **{field: value})
