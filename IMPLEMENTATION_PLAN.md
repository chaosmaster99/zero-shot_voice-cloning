# Implementation Plan and Status

| Phase | Scope | Status |
|---|---|---|
| 1 Environment | venv, backend/frontend scaffolding, dependency management (`requirements.txt`, `.env.example`) | done |
| 2 TTS backbone | F5-TTS selected as the single backbone (in-process API, model cached, seeded); the three-model benchmark of the spec was reduced to one practical backbone | done |
| 3 Zero-shot MVP | reference preprocessing/validation, `experiments/zero_shot/run_zero_shot.py`, automatic reference transcription (Whisper) | done |
| 4 Analysis & evaluation | waveform/STFT/mel, F0 (Praat AC), RMS, duration, pauses; ECAPA speaker similarity; Whisper WER | done |
| 5 Prosody controller | pitch, intonation range, energy, rate, pauses; emotion-inspired presets + custom; control accuracy (requested vs measured) | done |
| 6 Research experiments | `experiments/run_study.py`: single controls, combined, presets, ablation A–E, identity–prosody trade-off, figures, metadata | done (needs a reference set to produce results) |
| 7 FastAPI backend | `/health /models /presets /synthesize /prosody/apply /evaluate /analyze`, clean errors, size/type limits | done |
| 8 React frontend | upload/record, presets + custom sliders, baseline vs controlled, metrics, control accuracy, F0/energy/waveform/mel plots, prosody lab | done |
| 9 Tests & docs | signal, preset, control-accuracy, evaluation, preprocessing and API tests; README, `docs/PROSODY.md` | done |

## Not in scope / not done

* Training or fine-tuning any model (explicitly out of scope in `PROJECT_SPEC.md`).
* CosyVoice / Qwen3-TTS benchmark (one backbone only).
* Human MOS / SMOS evaluation.
* Word-level prosody control.
