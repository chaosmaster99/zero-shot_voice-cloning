# Controllable Zero-Shot Voice Cloning with Prosodic Manipulation

## Project overview

This project is a final-year speech-processing system for zero-shot voice cloning from a 5–10 second reference recording and controlled prosody modification. The goal is to study how prosodic manipulation affects speaker identity and speech intelligibility while preserving voice identity as much as possible.

## Research problem

The project explores the identity–prosody trade-off in zero-shot TTS: whether an unseen speaker can be cloned while controlled manipulations in pitch, energy, speaking rate, and pauses remain perceptible but do not destroy identity or intelligibility.

## Architecture

- Backend: FastAPI service for synthesis, evaluation, and analysis
- Audio processing: librosa, numpy, scipy, soundfile, torchaudio
- TTS layer: selected zero-shot backbone with a clean abstraction
- Evaluation: speaker similarity and WER pipeline
- Frontend: React + Vite + Tailwind interface

## Features

- Reference audio validation and preprocessing
- Zero-shot synthesis abstraction
- Prosody controller with pitch, energy, rate, and pause adjustments
- Speaker similarity scoring
- WER calculation
- Audio analysis utilities
- Experiments and reproducible metrics
- React UI for generation and comparison

## Installation

1. Open a terminal in the project root.
2. Create the virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it:

   - Windows PowerShell:

     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```

   - Windows cmd:

     ```cmd
     .\.venv\Scripts\activate.bat
     ```

4. Install Python dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. If the selected TTS package is required, install it in the same venv:

   ```bash
   python -m pip install f5-tts
   ```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

## Backend setup

```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## GPU requirements

This project is designed to run on CPU by default and can be configured for GPU if available. The environment used for this implementation reported no CUDA-enabled GPU in the active shell, so GPU execution is optional and should be enabled only when supported by the installed TTS backend and CUDA drivers.

## Running the project

### Health check

```bash
curl http://localhost:8000/health
```

### CLI zero-shot pass

```bash
python experiments/zero_shot/run_zero_shot.py --reference data/references/example.wav --text "Welcome to our Speech Processing project."
```

F5-TTS also requires the transcript of the reference recording. The web API accepts this as `reference_text`; an automatic ASR step should be added before claiming transcript-free reference support.

## Evaluation methodology

- Speaker similarity: cosine similarity on reference vs generated audio embeddings
- WER: ASR/transcription-based word error rate with normalized text comparison
- Prosody: F0, RMS energy, duration, and rate analysis from waveform statistics
- Control accuracy: compare requested value to measured value for pitch, energy, and rate changes

## Directory structure

```text
backend/
frontend/
models/
data/
experiments/
notebooks/
results/
tests/
```

## Limitations

- This repository contains the required project scaffold, evaluation logic, and runtime-safe architecture, but the selected zero-shot TTS package must be installed in the local environment for live synthesis to work.
- Some TTS backends are model-heavy and may require more memory or a compatible CUDA environment than this machine has available.
- Prosody control is implemented as a signal-processing layer and should be validated against the selected TTS model in the actual research pipeline.

## Ethics

Only use recordings for which consent has been obtained. Generated speech is synthetic and should not be used for unauthorized impersonation or deception.

## Future work

- Benchmark multiple TTS backbones against the same reference set
- Add richer model-native prosody control when supported by the selected model
- Expand experiment metadata and plot generation
- Add MOS and human evaluation workflows
- Improve the frontend with waveform and spectrogram plotting

## Project status

This repository is structured for the full research project and contains the ML-safe implementation skeleton, backend API, analysis modules, experiment entry points, and frontend template. The live TTS model must be installed and configured separately in the project environment before end-to-end synthesis can run.
