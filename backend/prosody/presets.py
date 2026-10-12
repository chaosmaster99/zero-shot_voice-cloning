"""Emotion presets for the DSP prosody controller.

Each preset is a combination of the eight signal-level controls. Directions follow the acoustic
correlates of vocal emotion reported in the literature (Murray & Arnott 1993; Burkhardt &
Sendlmeier 2000; Scherer 2003; Juslin & Laukka 2003):

  joy / happiness   higher F0, wider range, rising/lifted contours, louder, brighter, faster
  sadness           lower F0, narrow range, strong declination, softer, darker, slower, long pauses
  hot anger         raised F0, wide range, falling (assertive) contours, much louder, much brighter
  calm / relief     slightly lower F0, narrower range, softer, darker, unhurried
  excitement        very high F0, very wide range, rising contours, loud, bright, fast
  fear              high F0, F0 tremor (shaky voice), fast, slightly softer

They are starting points, not scientific definitions of emotion (see PROJECT_SPEC.md section 14);
the user can fine-tune every control ("custom").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.prosody.controller import NEUTRAL

NEUTRAL_PARAMS: Dict[str, float] = dict(NEUTRAL)
CONTROLS: Tuple[str, ...] = tuple(NEUTRAL_PARAMS)

# (min, max) accepted by the API / shown on the UI sliders.
LIMITS: Dict[str, Tuple[float, float]] = {
    "pitch": (-40.0, 60.0),
    "pitch_range": (0.2, 2.0),
    "pitch_slope": (-6.0, 6.0),
    "tremor": (0.0, 1.5),
    "energy": (-60.0, 80.0),
    "brightness": (-12.0, 12.0),
    "rate": (0.5, 2.0),
    "pause": (0.0, 3.0),
}

CUSTOM_KEY = "custom"


def _preset(**changes: float) -> Dict[str, float]:
    return {**NEUTRAL_PARAMS, **changes}


PRESETS: Dict[str, Dict[str, Any]] = {
    "neutral": {
        "label": "Neutral",
        "description": "No manipulation. The zero-shot baseline.",
        "params": dict(NEUTRAL_PARAMS),
    },
    "happy": {
        "label": "Happy",
        "description": "Higher, livelier melody with lifted phrase contours, a brighter and louder voice and a quicker pace.",
        "params": _preset(pitch=15.0, pitch_range=1.5, pitch_slope=1.5, energy=20.0, brightness=4.0, rate=1.1, pause=0.8),
    },
    "sad": {
        "label": "Sad",
        "description": "Low, flat and falling melody, a soft dark voice with a slight waver, slow pace and long pauses.",
        "params": _preset(pitch=-12.0, pitch_range=0.6, pitch_slope=-2.5, tremor=0.2, energy=-30.0, brightness=-6.0, rate=0.82, pause=1.6),
    },
    "angry": {
        "label": "Angry",
        "description": "Forceful: loud, tense and bright voice, wide falling (assertive) contours, clipped pauses.",
        "params": _preset(pitch=8.0, pitch_range=1.4, pitch_slope=-2.0, energy=40.0, brightness=8.0, rate=1.12, pause=0.7),
    },
    "calm": {
        "label": "Calm",
        "description": "Gently lowered, even melody, a soft and slightly dark voice, unhurried with relaxed pauses.",
        "params": _preset(pitch=-6.0, pitch_range=0.75, pitch_slope=-1.0, energy=-15.0, brightness=-3.0, rate=0.9, pause=1.3),
    },
    "excited": {
        "label": "Excited",
        "description": "Very high, animated and rising melody, loud bright voice, fast with minimal pauses.",
        "params": _preset(pitch=25.0, pitch_range=1.7, pitch_slope=2.0, energy=30.0, brightness=5.0, rate=1.2, pause=0.6),
    },
    "afraid": {
        "label": "Afraid",
        "description": "High, shaky voice (F0 tremor), hurried and slightly breathless.",
        "params": _preset(pitch=20.0, pitch_range=1.2, pitch_slope=0.5, tremor=0.6, energy=-10.0, brightness=2.0, rate=1.18, pause=0.9),
    },
    CUSTOM_KEY: {
        "label": "Custom",
        "description": "Set every control yourself.",
        "params": dict(NEUTRAL_PARAMS),
    },
}


def list_presets() -> List[Dict[str, Any]]:
    return [
        {"key": key, "label": spec["label"], "description": spec["description"], "params": dict(spec["params"])}
        for key, spec in PRESETS.items()
    ]


def validate_params(params: Dict[str, float]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for name, (low, high) in LIMITS.items():
        value = float(params[name])
        if not (low <= value <= high):
            raise ValueError(f"{name} must be between {low:g} and {high:g} (got {value:g}).")
        clean[name] = value
    return clean


def resolve_prosody(emotion: Optional[str] = None, **overrides: Optional[float]) -> Tuple[str, Dict[str, float]]:
    """Merge a preset with explicit overrides and validate the result.

    A preset provides the base values; any control passed explicitly (not None) overrides it.
    "custom" (or no emotion) starts from neutral.
    """
    key = (emotion or "neutral").strip().lower()
    if key not in PRESETS:
        raise ValueError(f"Unknown emotion preset '{emotion}'. Available: {', '.join(PRESETS)}.")
    unknown = set(overrides) - set(CONTROLS)
    if unknown:
        raise ValueError(f"Unknown prosody control(s): {', '.join(sorted(unknown))}.")

    params = dict(PRESETS[key]["params"])
    for name, value in overrides.items():
        if value is not None:
            params[name] = float(value)
    return key, validate_params(params)
