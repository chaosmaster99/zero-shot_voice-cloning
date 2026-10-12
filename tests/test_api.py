"""API tests with a *test-double* TTS engine and fake evaluation models.

The double only exists so the HTTP/prosody/evaluation plumbing can be tested without downloading
multi-GB checkpoints; it is never used by the application itself.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tests.conftest import SR, synth_speechlike


class StubEngine:
    name = "stub"

    def availability(self):
        return True, "stub engine"

    def synthesize(self, reference_audio, text, output_path, *, reference_text="", seed=None):
        sf.write(str(output_path), synth_speechlike(f0=118.0), SR)
        return {
            "output_path": str(output_path),
            "engine": "stub",
            "model": "stub",
            "seed": seed or 0,
            "nfe_steps": 0,
            "reference_text": reference_text or "auto transcript",
            "reference_text_source": "user" if reference_text else "whisper",
        }


@pytest.fixture()
def client(monkeypatch):
    import backend.core.pipeline as pipeline
    import backend.evaluation.asr as asr
    import backend.evaluation.speaker_similarity as speaker
    from backend.main import app

    monkeypatch.setattr(pipeline, "get_tts_engine", lambda *a, **k: StubEngine())
    monkeypatch.setattr(speaker, "speaker_embedding", lambda audio, sr: np.array([1.0, 0.0]))
    monkeypatch.setattr(speaker, "embedding_similarity", lambda a, b: 0.91)
    monkeypatch.setattr(asr, "transcribe", lambda audio, sr: "welcome to our speech processing project")
    monkeypatch.setattr("backend.api.routes.health.get_tts_engine", lambda *a, **k: StubEngine())
    return TestClient(app)


def _wav_bytes(seconds_per_phrase: float = 2.2, sample_rate: int = SR) -> bytes:
    buffer = io.BytesIO()
    audio = synth_speechlike(phrases=2, phrase_seconds=seconds_per_phrase, pause_seconds=0.3, sample_rate=sample_rate)
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def _post_synth(client, **form):
    data = {"text": "Welcome to our speech processing project.", "reference_text": "hello there", **form}
    return client.post("/synthesize", data=data, files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")})


def test_health_and_presets(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["tts_available"] is True

    payload = client.get("/presets").json()
    keys = [p["key"] for p in payload["presets"]]
    assert {"neutral", "happy", "sad", "angry", "calm", "excited", "custom"} <= set(keys)
    assert payload["limits"]["rate"] == {"min": 0.5, "max": 2.0}

    models = client.get("/models").json()
    assert models["pretrained_only"] is True


def test_synthesize_preset_returns_audio_metrics_and_control_accuracy(client):
    response = _post_synth(client, emotion="happy")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["emotion"] == "happy"
    assert body["prosody"]["rate"] == pytest.approx(1.1)
    assert {"reference", "baseline", "controlled"} <= set(body["urls"])
    for url in body["urls"].values():
        assert client.get(url).status_code == 200

    assert body["metrics"]["speaker_similarity"] == pytest.approx(0.91)
    assert body["metrics"]["wer"] == pytest.approx(0.0)
    assert set(body["control_accuracy"]) == {"pitch", "pitch_range", "pitch_slope", "tremor", "energy", "brightness", "rate", "pause"}
    assert body["control_accuracy"]["pitch"]["measured"] > 5.0  # happy raises pitch
    assert body["control_accuracy"]["rate"]["measured"] > 1.0
    assert "contours" in body["evaluation"]["clips"]["controlled"]["prosody"]
    assert body["analysis"]["controlled"]["mel"]["frames"] > 0
    assert body["timings"]["rtf"] >= 0


def test_sad_slows_and_lowers(client):
    body = _post_synth(client, emotion="sad").json()
    assert body["control_accuracy"]["pitch"]["measured"] < -5.0
    assert body["control_accuracy"]["rate"]["measured"] < 1.0
    assert body["control_accuracy"]["energy"]["measured"] < -10.0


def test_custom_values_are_applied_exactly(client):
    body = _post_synth(client, emotion="custom", pitch="12", energy="-15", rate="1.25", pause="1", pitch_range="1").json()
    assert body["emotion"] == "custom"
    assert body["prosody"] == {"pitch": 12.0, "pitch_range": 1.0, "pitch_slope": 0.0, "tremor": 0.0, "energy": -15.0,
                               "brightness": 0.0, "rate": 1.25, "pause": 1.0}
    accuracy = body["control_accuracy"]
    assert abs(accuracy["pitch"]["measured"] - 12) < 4
    assert abs(accuracy["energy"]["measured"] + 15) < 4
    assert abs(accuracy["rate"]["measured"] - 1.25) < 0.08


def test_invalid_inputs_return_clean_400s(client):
    assert _post_synth(client, text="   ").status_code == 400
    bad_preset = _post_synth(client, emotion="furious")
    assert bad_preset.status_code == 400 and "Unknown emotion" in bad_preset.json()["detail"]
    out_of_range = _post_synth(client, emotion="custom", rate="9")
    assert out_of_range.status_code == 400 and "rate" in out_of_range.json()["detail"]

    short = client.post(
        "/synthesize",
        data={"text": "hi", "reference_text": "hi"},
        files={"reference_audio": ("short.wav", _wav_bytes(seconds_per_phrase=0.4), "audio/wav")},
    )
    assert short.status_code == 400 and "at least" in short.json()["detail"]

    junk = client.post(
        "/synthesize",
        data={"text": "hi"},
        files={"reference_audio": ("x.wav", b"not audio at all", "audio/wav")},
    )
    assert junk.status_code == 400

    wrong_type = client.post(
        "/synthesize",
        data={"text": "hi"},
        files={"reference_audio": ("x.exe", b"MZ", "application/octet-stream")},
    )
    assert wrong_type.status_code == 400


def test_errors_never_leak_tracebacks(client, monkeypatch):
    import backend.core.pipeline as pipeline

    class Exploding(StubEngine):
        def synthesize(self, *a, **k):
            raise RuntimeError("secret internal path C:\\models\\x.pt")

    monkeypatch.setattr(pipeline, "get_tts_engine", lambda *a, **k: Exploding())
    response = _post_synth(client)
    assert response.status_code == 500
    assert "secret" not in response.text and "Traceback" not in response.text


def test_tts_unavailable_is_503(client, monkeypatch):
    import backend.core.pipeline as pipeline
    from backend.core.tts_engine import TTSUnavailable

    class Missing(StubEngine):
        def synthesize(self, *a, **k):
            raise TTSUnavailable("f5-tts is not installed")

    monkeypatch.setattr(pipeline, "get_tts_engine", lambda *a, **k: Missing())
    assert _post_synth(client).status_code == 503


def test_prosody_lab_applies_preset_to_any_recording(client):
    response = client.post(
        "/prosody/apply",
        data={"emotion": "excited"},
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["emotion"] == "excited"
    assert body["control_accuracy"]["pitch"]["measured"] > 8.0
    assert body["control_accuracy"]["rate"]["measured"] > 1.1
    assert set(body["urls"]) == {"baseline", "controlled"}


def test_evaluate_and_analyze_endpoints(client):
    files = {
        "reference_audio": ("ref.wav", _wav_bytes(), "audio/wav"),
        "generated_audio": ("gen.wav", _wav_bytes(), "audio/wav"),
    }
    evaluated = client.post("/evaluate", data={"target_text": "welcome to our speech processing project"}, files=files).json()
    assert evaluated["speaker_similarity"] == pytest.approx(0.91)
    assert evaluated["wer"] == pytest.approx(0.0)

    analyzed = client.post("/analyze", files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")}).json()
    assert analyzed["prosody"]["f0_median"] > 50
    assert analyzed["prosody"]["contours"]["f0"]


def test_generated_audio_files_exist_and_uploads_are_removed(client):
    from backend.config import OUTPUT_DIRECTORY

    body = _post_synth(client).json()
    for name in body["urls"].values():
        assert (Path(OUTPUT_DIRECTORY) / Path(name).name).exists()
    assert not list(Path(OUTPUT_DIRECTORY).glob(f"{body['request_id']}_*_upload.*"))
