# Implementation Plan

## Phase 1 — Environment and project baseline
- Confirm OS, Python, Node, GPU/CUDA availability
- Create `.venv` and install Python dependencies locally
- Create repository structure and initial config files
- Verify backend/frontend can run independently

## Phase 2 — TTS benchmark
- Install and test candidate backbones: CosyVoice, Qwen3-TTS, F5-TTS
- Run a lightweight benchmark on a shared reference text/audio pair
- Record success/failure, latency, similarity, WER, VRAM, notes
- Select one practical backbone for the production pipeline

## Phase 3 — Zero-shot MVP
- Implement audio preprocessing and validation
- Add a CLI/evaluation script for synthesis from a reference audio and text
- Verify generation produces a valid WAV file using the selected TTS engine

## Phase 4 — Audio analysis and evaluation
- Implement signal-processing modules: waveform, STFT, spectrogram, mel, F0, energy, duration, pause detection
- Implement speaker similarity via a consistent embedding model
- Implement ASR-based WER with fallbacks for failure

## Phase 5 — Prosody controller
- Add pitch, energy, and speaking-rate controls sequentially
- Validate actual change against requested values
- Add pause manipulation if the selected model supports it reliably
- Build the baseline vs controlled comparison flow

## Phase 6 — Research experiments
- Implement ablation and identity–prosody trade-off scripts
- Save metadata, metrics, and generated audio paths to reproducible records
- Generate plots from real measurements

## Phase 7 — FastAPI backend
- Expose `/health`, `/models`, `/synthesize`, `/evaluate`, and `/analyze`
- Validate audio/text inputs and return structured errors
- Ensure synthetic outputs and evaluation are computed from actual model results

## Phase 8 — React frontend
- Build a polished UI with upload, playback, text input, controls, and comparison panels
- Connect frontend to backend API and visualize metrics

## Phase 9 — Testing and documentation
- Add focused tests for audio, evaluation, and API behavior
- Update README with setup, usage, architecture, and limitations
- Verify the full stack runs end-to-end with a single example
