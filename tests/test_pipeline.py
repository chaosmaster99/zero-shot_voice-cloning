"""Pipeline-level behaviour with a test-double TTS engine (no checkpoints needed)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from backend.core.pipeline import synthesize_baseline, synthesize_with_prosody
from backend.evaluation.prosody_metrics import compute_prosody_metrics
from tests.conftest import SR, synth_speechlike


class OnePhraseEngine:
    """Returns one continuous phrase per call, with ragged silence at both ends."""

    name = "double"

    def __init__(self):
        self.calls = []

    def availability(self):
        return True, "double"

    def synthesize(self, reference_audio, text, output_path, *, reference_text="", seed=None):
        self.calls.append(text)
        audio = synth_speechlike(phrases=1, phrase_seconds=1.6, lead=0.3)
        sf.write(str(output_path), audio, SR)
        return {"output_path": str(output_path), "engine": "double", "model": "double", "seed": seed or 0,
                "nfe_steps": 0, "reference_text": reference_text or "transcribed", "reference_text_source": "double"}


def test_baseline_is_generated_sentence_by_sentence_with_explicit_gaps(tmp_path):
    engine = OnePhraseEngine()
    out = tmp_path / "baseline.wav"
    meta = synthesize_baseline(engine, tmp_path / "ref.wav", "First sentence is here. Second sentence follows. Third one ends it.",
                               out, reference_text="", sentence_pause=0.4)

    assert len(engine.calls) == 3 and meta["sentences"] == engine.calls
    audio, sr = sf.read(out)
    metrics = compute_prosody_metrics(audio.astype(np.float32), sr)
    assert metrics["pause_count"] == 2
    assert abs(metrics["pause_total"] - 0.8) < 0.15
    assert float(np.max(np.abs(audio))) == pytest.approx(0.5, abs=0.01)


def test_reference_is_transcribed_once_not_per_sentence(tmp_path):
    seen = []

    class Recorder(OnePhraseEngine):
        def synthesize(self, reference_audio, text, output_path, *, reference_text="", seed=None):
            seen.append(reference_text)
            return super().synthesize(reference_audio, text, output_path, reference_text=reference_text, seed=seed)

    meta = synthesize_baseline(Recorder(), tmp_path / "ref.wav", "One sentence goes here. Another sentence goes there.",
                               tmp_path / "b.wav", reference_text="")
    assert seen == ["", "transcribed"]
    assert meta["reference_text_source"] == "double"  # the source of the *first* (transcribing) call is kept


def test_pause_control_scales_the_explicit_sentence_gaps(tmp_path, monkeypatch):
    import backend.core.pipeline as pipeline
    import backend.evaluation.asr as asr
    import backend.evaluation.speaker_similarity as speaker

    monkeypatch.setattr(pipeline, "get_tts_engine", lambda *a, **k: OnePhraseEngine())
    monkeypatch.setattr(speaker, "speaker_embedding", lambda audio, sr: np.array([1.0, 0.0]))
    monkeypatch.setattr(speaker, "embedding_similarity", lambda a, b: 0.9)
    monkeypatch.setattr(asr, "transcribe", lambda audio, sr: "first sentence is here second sentence follows")

    reference = tmp_path / "ref.wav"
    sf.write(str(reference), synth_speechlike(phrases=2, phrase_seconds=2.4, pause_seconds=0.3), SR)

    result = synthesize_with_prosody(
        request_id="t1", reference_path=reference, reference_text="hi", text="First sentence is here. Second sentence follows.",
        output_dir=tmp_path, emotion="custom", overrides={"pause": 2.0},
    )
    accuracy = result["control_accuracy"]["pause"]
    assert accuracy["measured"] is not None
    assert abs(accuracy["measured"] - 2.0) < 0.15
    assert result["run"]["sentences"] == ["First sentence is here.", "Second sentence follows."]
