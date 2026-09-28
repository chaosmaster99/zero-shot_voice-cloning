# Controllable Zero-Shot Voice Cloning with Prosodic Manipulation

## 1. Project Overview

### Course
22AIE450 – Speech Processing

### Project Type
Final-Year Semester Project

### Project Title

**Controllable Zero-Shot Voice Cloning with Prosodic Manipulation**

### Core Objective

Build an interactive system capable of cloning an unseen speaker's voice from only **5–10 seconds of reference speech** and generating arbitrary English text in that speaker's voice.

The system must additionally allow controlled manipulation of speech delivery through:

- Pitch
- Energy
- Speaking rate
- Pauses

The primary research objective is to investigate the relationship between:

**Speaker Identity ↔ Prosodic Manipulation ↔ Speech Intelligibility**

The project should determine how strongly speech delivery can be modified while preserving:

1. Speaker identity
2. Speech intelligibility
3. Naturalness

The project will use pretrained/open-weight zero-shot TTS models rather than training a complete TTS model from scratch.

---

# 2. Research Problem

Traditional TTS systems generally focus on generating natural speech from text.

Modern zero-shot TTS systems can additionally reproduce an unseen speaker's voice from a short reference recording.

However, an important question remains:

> Can the identity of an unseen speaker be preserved while independently manipulating the prosodic characteristics of the generated speech?

This project investigates that problem.

---

# 3. Research Questions

## RQ1 — Zero-Shot Speaker Cloning

Can a pretrained zero-shot TTS system reproduce the identity of an unseen speaker using only 5–10 seconds of reference speech?

## RQ2 — Prosodic Control

Can explicit manipulation of pitch, energy, speaking rate and pauses produce predictable changes in the synthesized speech?

## RQ3 — Identity–Prosody Trade-Off

How does increasing prosodic manipulation affect:

- Speaker similarity
- Speech intelligibility
- Naturalness

RQ3 is the primary research contribution.

---

# 4. Project Contribution

The project is NOT claiming to invent a new TTS architecture.

The contribution is an experimental and system-level framework for:

1. Zero-shot voice cloning
2. Explicit prosody manipulation
3. Quantitative evaluation of prosody control
4. Measurement of speaker identity preservation
5. Analysis of the identity–prosody trade-off

The central concept is:

```text
WHO is speaking?
        ↓
Speaker Conditioning
        ↓
Speaker Identity


HOW are they speaking?
        ↓
Prosodic Manipulation
        ↓
Pitch / Energy / Rate / Pause
```

The goal is to modify:

**HOW**

while preserving:

**WHO**

---

# 5. Scope

## MUST IMPLEMENT

### Zero-Shot Voice Cloning

- 5–10 second reference audio
- Unseen speakers
- Arbitrary English text
- Pretrained/open-weight zero-shot TTS
- No full TTS training from scratch

### Prosody Control

- Pitch
- Energy
- Speaking rate
- Pause manipulation where technically feasible

### Evaluation

- Speaker embedding similarity
- WER
- F0
- Energy
- Duration / speaking rate
- Pause duration
- Real-Time Factor
- Optional human evaluation

### Research

- Baseline vs proposed system
- Control accuracy
- Ablation study
- Identity–prosody trade-off experiment

### Application

- Modern web UI
- Reference audio upload/recording
- Text input
- Prosody controls
- Audio playback
- Metrics display
- Comparison of baseline and controlled outputs

---

# 6. OUT OF SCOPE FOR INITIAL IMPLEMENTATION

Do NOT make these mandatory:

- Training a TTS model from scratch
- Training a speaker encoder
- Multilingual TTS
- Word-level prosody control
- Learned emotion model
- Production-scale deployment
- Voice authentication
- Complex distributed infrastructure
- Advanced watermarking
- Multiple simultaneous TTS backbones in production

These may be added only after the core project is completely functional.

---

# 7. TTS Backbone Strategy

Initially benchmark:

1. CosyVoice
2. Qwen3-TTS
3. F5-TTS

The benchmark must be lightweight.

The purpose is to select ONE practical primary backbone.

Do not build the entire project around all three models simultaneously.

Selection criteria:

- Zero-shot cloning quality
- Speaker similarity
- WER
- Naturalness
- Inference latency
- VRAM requirement
- Reference audio requirements
- Availability of useful control mechanisms
- Ease of integration

After benchmarking:

```text
CosyVoice
Qwen3-TTS
F5-TTS
       ↓
Benchmark
       ↓
Select ONE
       ↓
Final System
```

The selected model becomes the main TTS backbone.

---

# 8. Zero-Shot Requirement

The system must support:

```text
5–10 sec reference audio
        ↓
Speaker conditioning
        ↓
Target text
        ↓
Generated speech
```

The evaluation speakers must remain unseen during any training/adaptation used for the experiment.

Do not claim zero-shot performance if the evaluation speaker was used for model adaptation.

---

# 9. Reference Audio Processing

Input:

```text
5–10 second speech recording
```

Pipeline:

```text
Raw Audio
    ↓
Mono Conversion
    ↓
Resampling
    ↓
Amplitude Normalization
    ↓
Voice Activity Detection
    ↓
Silence Cleanup
    ↓
Clean Reference
```

The system should preferably work with clean speech.

The reference should ideally contain relatively stable speech and limited background noise.

Do not require the reference to be perfectly neutral.

---

# 10. Speaker Conditioning

The implementation must respect the selected TTS architecture.

Do NOT assume every model uses an explicit external speaker encoder.

Conceptually:

```text
Reference Audio
       ↓
TTS Speaker Conditioning
       ↓
Speaker Representation
       ↓
Zero-Shot TTS
```

A separate speaker encoder may be used for evaluation.

For example:

```text
Reference Audio
      ↓
Speaker Encoder
      ↓
Reference Embedding

Generated Audio
      ↓
Same Speaker Encoder
      ↓
Generated Embedding

Reference Embedding
        ↕
Cosine Similarity
        ↕
Generated Embedding
```

---

# 11. Text Processing

Input:

```text
Welcome to our Speech Processing project.
```

Pipeline:

```text
Raw Text
   ↓
Text Normalization
   ↓
Model Tokenization / Native Text Processing
   ↓
TTS
```

Do NOT rebuild a G2P or phoneme system if the selected TTS model already provides an effective text frontend.

---

# 12. Prosody Controller

The custom project layer is the prosody controller.

Core controls:

### Pitch

Example:

```text
-20%
-10%
Normal
+10%
+20%
```

### Energy

Example:

```text
-20%
Normal
+20%
```

### Speaking Rate

Example:

```text
0.8x
1.0x
1.2x
```

### Pause

Allow sentence-level pause adjustment if technically reliable.

Do NOT expose duration as a completely independent control initially because speaking rate and total duration are strongly coupled.

Duration should primarily be measured as an output.

---

# 13. Prosody Implementation Strategy

Use the following fallback hierarchy.

## Level 1 — Native Model Controls

If the selected TTS model supports:

- Style
- Emotion
- Instruction
- Prosody controls
- Expressive tokens

use them.

## Level 2 — Structured Controls

If supported:

- Pause tokens
- Emphasis
- Style tokens
- Other model-specific controls

## Level 3 — Post-Synthesis Processing

If direct control is insufficient:

```text
TTS
 ↓
Generated Speech
 ↓
Prosody Processor
 ↓
Final Speech
```

The processor may modify:

- F0 / pitch
- Energy
- Speaking rate
- Pauses

Use established signal-processing techniques rather than arbitrary waveform manipulation.

## Level 4 — Fine-Tuning

Fine-tuning is optional.

Only attempt it if:

1. The pretrained system works
2. Native/post-processing controls are insufficient
3. Adequate data and compute are available
4. The project timeline permits it

Never make fine-tuning a dependency for the MVP.

---

# 14. Emotion Presets

Emotion is NOT a primary scientific component.

Instead, implement optional:

**Emotion-inspired prosody presets**

Examples:

```text
Neutral
Excited
Calm
Sad
```

These presets configure combinations of:

- Pitch
- Energy
- Rate
- Pauses

Example:

```text
Excited
    Pitch   ↑
    Energy  ↑
    Rate    ↑
    Pauses  ↓
```

These presets must NOT be presented as universal scientific definitions of emotion.

They are high-level prosody presets.

Manual controls must remain available.

---

# 15. Baseline

The baseline system is:

```text
Reference Audio
       ↓
Zero-Shot TTS
       ↓
Generated Speech
```

No additional prosody manipulation.

---

# 16. Proposed System

The proposed system is:

```text
Reference Audio
       ↓
Zero-Shot TTS
       ↓
Baseline Speech
       ↓
Prosody Controller
       ↓
Controlled Speech
```

The baseline and controlled output must use:

- Same reference speaker
- Same target text
- Same TTS backbone
- Same evaluation pipeline

---

# 17. Evaluation Metrics

## 17.1 Speaker Identity

Use speaker embeddings.

Primary metric:

**Cosine Similarity / Speaker Embedding Cosine Similarity**

Compare:

```text
Reference Speaker
        ↕
Generated Speaker
```

Higher similarity indicates stronger preservation of speaker characteristics.

Do not interpret a single threshold as universal speaker identity verification.

---

# 18. Speech Intelligibility

Use:

**Word Error Rate (WER)**

Pipeline:

```text
Target Text
    ↕
Generated Audio
    ↓
ASR
    ↓
WER
```

WER measures content preservation.

---

# 19. Pitch Evaluation

Extract F0 from:

1. Reference
2. Baseline
3. Controlled output

Measure:

- Mean F0
- Median F0
- Relative F0 change

Example:

```text
Requested:
+20% pitch

Measured:
+18.7%
```

Calculate control error.

---

# 20. Energy Evaluation

Measure RMS energy.

Compare:

```text
Baseline Energy
       ↓
Controlled Energy
```

Calculate relative change and control error.

---

# 21. Speaking Rate Evaluation

Measure:

- Total duration
- Speech duration
- Words per second where appropriate

Compare requested rate with actual generated rate.

---

# 22. Pause Evaluation

If pause control is implemented:

Measure:

- Number of pauses
- Pause duration
- Relative pause change

---

# 23. Control Accuracy

For each control:

```text
Requested Value
       ↓
Generated Measurement
       ↓
Absolute Error
```

Examples:

```text
Pitch Control Error
Energy Control Error
Rate Control Error
Pause Control Error
```

This is an important part of the research contribution.

---

# 24. Naturalness

If time and resources permit, perform human evaluation.

Use:

**MOS — Mean Opinion Score**

Participants rate naturalness of generated speech.

Do not make MOS mandatory for the first MVP.

---

# 25. Speaker Similarity Human Evaluation

Optional:

**SMOS / speaker similarity rating**

Participants hear:

1. Reference voice
2. Generated voice

and rate perceived speaker similarity.

---

# 26. System Performance

Measure:

- Inference latency
- Real-Time Factor
- GPU VRAM usage
- Reference processing time

These metrics should be reported for the final selected backbone.

---

# 27. Required Experiments

## Experiment 1 — Zero-Shot Voice Cloning

For multiple unseen speakers:

```text
5–10 sec reference
        ↓
Target sentence
        ↓
Generated speech
```

Measure:

- Speaker similarity
- WER
- Optional MOS

---

## Experiment 2 — Pitch Control

Test:

```text
-20%
-10%
Normal
+10%
+20%
```

Measure:

- Actual F0
- Control error
- Speaker similarity
- WER

---

## Experiment 3 — Energy Control

Test:

```text
-20%
Normal
+20%
```

Measure:

- RMS energy
- Control error
- Speaker similarity
- WER

---

## Experiment 4 — Speaking Rate

Test:

```text
0.8x
1.0x
1.2x
```

Measure:

- Duration
- Speaking rate
- Control error
- Speaker similarity
- WER

---

## Experiment 5 — Combined Prosody

Example:

```text
Pitch   +15%
Energy  +10%
Rate    +10%
```

Measure whether multiple controls interact predictably.

---

# 28. Main Research Experiment

## Identity–Prosody Trade-Off

This is the central experiment.

Define increasing control strength:

```text
Level 0
Normal

Level 1
Mild modification

Level 2
Moderate modification

Level 3
Strong modification
```

For every level measure:

- Speaker similarity
- WER
- F0
- Energy
- Duration/rate
- Control error
- Naturalness where possible

Analyze:

```text
Prosody Modification Strength
            ↓
     Speaker Similarity
            ↓
      Speech Quality
```

The project should NOT assume in advance what the relationship will be.

The experiment is intended to discover and quantify the relationship.

---

# 29. Ablation Study

Perform an ablation where feasible.

Example:

```text
A. Baseline

B. Baseline + Pitch

C. Baseline + Pitch + Energy

D. Baseline + Pitch + Energy + Rate

E. Full Proposed System
```

Compare:

- Speaker similarity
- WER
- Prosody control accuracy

This determines the contribution of each control component.

---

# 30. Dataset Strategy

Use publicly available speech datasets suitable for zero-shot evaluation.

Potential datasets:

- LibriTTS
- VCTK

The exact dataset should be finalized after checking:

- Licensing
- Audio quality
- Speaker count
- Compatibility with the selected TTS model
- Computational requirements

Use multiple speakers.

Do not evaluate only on the team/demo speaker.

---

# 31. Evaluation Protocol

Recommended starting protocol:

```text
5–10 unseen speakers
        ×
3–5 fixed target sentences
        ×
multiple control conditions
```

Increase the number only if computational resources permit.

Keep target sentences fixed across experiments when comparing conditions.

This ensures that differences are attributable to the system/control rather than changing text.

---

# 32. Speech Processing Analysis

For reference, baseline and controlled audio, generate:

### Time domain

- Waveform

### Time-frequency domain

- STFT
- Spectrogram

### Mel representation

- Mel-spectrogram

### Prosodic features

- F0
- RMS energy
- Duration
- Pause statistics

The UI/report should be able to visually compare:

```text
Reference
Baseline
Controlled
```

This explicitly connects the project to Speech Processing concepts.

---

# 33. Application Architecture

## Frontend

Use:

- React
- Vite
- Tailwind CSS

The frontend should provide a polished modern UI.

Required functionality:

- Upload reference audio
- Record reference audio if feasible
- Audio playback
- Target text input
- Emotion preset
- Pitch slider
- Energy slider
- Speaking-rate slider
- Pause control where supported
- Generate button
- Loading/progress state
- Baseline vs controlled output
- Audio waveform/player
- Metrics dashboard
- Prosody visualizations

---

# 34. Backend

Use:

- Python
- FastAPI

Responsibilities:

- Audio upload
- Audio validation
- Reference preprocessing
- TTS inference
- Prosody processing
- Audio generation
- Evaluation
- Metrics calculation
- Result management

Do not put model logic directly inside frontend code.

---

# 35. ML Layer

Use:

- Python
- PyTorch
- torchaudio
- librosa where appropriate
- Hugging Face where applicable

Components:

```text
Model Manager
Reference Processor
Speaker Conditioning
TTS Inference
Prosody Processor
Audio Processor
Evaluation Engine
Metrics Engine
```

---

# 36. Recommended Directory Structure

```text
controllable-voice-cloning/
│
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── backend/
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── synthesis.py
│   │   │   ├── evaluation.py
│   │   │   └── models.py
│   │   │
│   │   └── schemas/
│   │       ├── synthesis.py
│   │       └── evaluation.py
│   │
│   ├── core/
│   │   ├── model_manager.py
│   │   ├── tts_engine.py
│   │   └── device.py
│   │
│   ├── audio/
│   │   ├── preprocessing.py
│   │   ├── postprocessing.py
│   │   ├── f0.py
│   │   ├── energy.py
│   │   ├── duration.py
│   │   └── pauses.py
│   │
│   ├── prosody/
│   │   ├── controller.py
│   │   ├── pitch.py
│   │   ├── energy.py
│   │   ├── rate.py
│   │   ├── pauses.py
│   │   └── presets.py
│   │
│   ├── evaluation/
│   │   ├── speaker_similarity.py
│   │   ├── wer.py
│   │   ├── prosody_metrics.py
│   │   ├── naturalness.py
│   │   └── performance.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── file_utils.py
│       └── audio_utils.py
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   │
│   └── src/
│       ├── components/
│       │   ├── AudioRecorder.jsx
│       │   ├── AudioUploader.jsx
│       │   ├── TextInput.jsx
│       │   ├── ProsodyControls.jsx
│       │   ├── EmotionSelector.jsx
│       │   ├── AudioPlayer.jsx
│       │   ├── MetricsCard.jsx
│       │   ├── Waveform.jsx
│       │   └── ComparisonPanel.jsx
│       │
│       ├── pages/
│       │   └── Home.jsx
│       │
│       ├── services/
│       │   └── api.js
│       │
│       ├── App.jsx
│       ├── main.jsx
│       └── index.css
│
├── models/
│   └── README.md
│
├── data/
│   ├── references/
│   ├── generated/
│   └── evaluation/
│
├── experiments/
│   ├── benchmark/
│   ├── zero_shot/
│   ├── pitch/
│   ├── energy/
│   ├── rate/
│   ├── combined/
│   ├── ablation/
│   └── identity_prosody/
│
├── notebooks/
│   ├── 01_audio_analysis.ipynb
│   ├── 02_model_benchmark.ipynb
│   ├── 03_prosody_analysis.ipynb
│   └── 04_identity_prosody_tradeoff.ipynb
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── reports/
│
└── tests/
    ├── test_audio.py
    ├── test_prosody.py
    ├── test_evaluation.py
    └── test_api.py
```

---

# 37. Virtual Environment

The project MUST use a Python virtual environment.

Recommended:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Never install project dependencies globally.

---

# 38. Dependency Management

Maintain:

```text
requirements.txt
```

and keep dependencies reproducible.

Document:

- Python version
- PyTorch version
- CUDA version
- GPU used
- TTS model version
- OS

---

# 39. Development Order

Follow this order.

## Phase 0 — Environment

Set up:

- Git repository
- Python virtual environment
- Backend
- Frontend
- Basic FastAPI endpoint
- Basic React application

---

## Phase 1 — TTS Benchmark

Implement minimal inference for:

- CosyVoice
- Qwen3-TTS
- F5-TTS

Use a small test set.

Do not build the UI yet.

Choose the practical backbone.

---

## Phase 2 — Zero-Shot MVP

Implement:

```text
Reference
 ↓
TTS
 ↓
Text
 ↓
Generated Audio
```

Success criterion:

The system reliably produces recognizable speech in the reference speaker's voice.

---

## Phase 3 — Evaluation Pipeline

Implement:

- Speaker similarity
- WER
- F0
- Energy
- Duration
- RTF

Create automated evaluation scripts.

---

## Phase 4 — Prosody Control

Implement:

- Pitch
- Energy
- Speed
- Pause

One at a time.

Do not implement all controls simultaneously.

---

## Phase 5 — Research Experiments

Implement:

- Baseline
- Control experiments
- Combined control
- Ablation
- Identity–prosody trade-off

---

## Phase 6 — Application

Build the polished React application.

---

## Phase 7 — Human Evaluation

If time permits:

- MOS
- Speaker similarity ratings

---

## Phase 8 — Final Evaluation

Generate:

- Tables
- Graphs
- Spectrograms
- F0 plots
- Speaker similarity plots
- Control-error plots
- Identity–prosody trade-off plots

---

# 40. UI Design Requirements

The interface should look like a modern AI application rather than a basic college CRUD application.

Design principles:

- Dark/light professional theme
- Clean typography
- Spacious layout
- Card-based sections
- Clear hierarchy
- Responsive design
- Smooth loading states
- Audio waveform visualization
- Clear comparison between baseline and controlled speech

Suggested layout:

```text
┌──────────────────────────────────────────────────────┐
│ CONTROLLABLE VOICE                                   │
│ Zero-Shot Voice Cloning & Prosody Studio             │
├──────────────────────────────────────────────────────┤
│                                                      │
│ REFERENCE VOICE                                      │
│                                                      │
│ [ Upload Audio ] [ Record ]                          │
│                                                      │
│ ───────────── Audio Waveform ─────────────            │
│                                                      │
├──────────────────────────────────────────────────────┤
│ TARGET TEXT                                          │
│                                                      │
│ ┌──────────────────────────────────────────────────┐ │
│ │ Enter the text you want the cloned voice to say │ │
│ └──────────────────────────────────────────────────┘ │
│                                                      │
├──────────────────────────────────────────────────────┤
│ PROSODY CONTROL                                      │
│                                                      │
│ Emotion       [ Neutral ▼ ]                          │
│                                                      │
│ Pitch         ─────────●────────                     │
│ Energy        ───────●──────────                     │
│ Speed         ─────────●────────                     │
│ Pause         ────────●────────                     │
│                                                      │
│              [ Generate Speech ]                     │
│                                                      │
├──────────────────────────────────────────────────────┤
│ RESULTS                                              │
│                                                      │
│ ┌──────────────────┐ ┌──────────────────┐            │
│ │ BASELINE         │ │ CONTROLLED       │            │
│ │                  │ │                  │            │
│ │ ▶ Audio          │ │ ▶ Audio          │            │
│ └──────────────────┘ └──────────────────┘            │
│                                                      │
├──────────────────────────────────────────────────────┤
│ EVALUATION                                           │
│                                                      │
│ Speaker Similarity   0.XX                            │
│ WER                  X.X%                            │
│ F0 Change            +XX%                            │
│ Energy Change        +XX%                            │
│ Speaking Rate        X.Xx                            │
│                                                      │
├──────────────────────────────────────────────────────┤
│ SPEECH ANALYSIS                                      │
│                                                      │
│ [Waveform]                                           │
│ [Spectrogram]                                        │
│ [F0 Curve]                                           │
│ [Energy Curve]                                       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

# 41. Error Handling

The application must gracefully handle:

- Invalid audio format
- Audio too short
- Audio too long
- No speech detected
- Excessive background noise
- Empty text
- Model loading failure
- GPU memory failure
- TTS inference failure
- Evaluation failure

Never expose raw Python stack traces to the frontend.

---

# 42. Logging

Backend should log:

- Request ID
- Processing time
- Model used
- Reference duration
- Target text length
- Prosody parameters
- Generation time
- Evaluation time
- Errors

Do not log sensitive audio data unnecessarily.

---

# 43. Reproducibility

Every experiment must record:

```text
Model
Model version
Dataset
Speaker ID
Reference duration
Target text
Control parameters
Random seed where applicable
Hardware
Software versions
Metrics
```

---

# 44. Ethical Requirements

Only use voices where permission/consent has been obtained for demonstration.

The application must clearly state:

> Generated speech is synthetic and should not be used for unauthorized impersonation or deception.

Do not use public figures' voices for the primary demonstration.

Optional future enhancement:

- Synthetic audio watermarking

---

# 45. Success Criteria

The project is considered successful if it achieves:

### Minimum

- Working zero-shot cloning
- 5–10 sec reference
- Multiple unseen speakers
- Arbitrary text
- Baseline generation
- At least 2 reliable prosody controls
- Speaker similarity evaluation
- WER evaluation
- Speech signal analysis
- Working web interface

### Strong Final Version

Everything above plus:

- Pitch
- Energy
- Rate
- Pause
- Control accuracy
- Ablation
- Human evaluation
- Identity–prosody trade-off analysis
- Professional UI

---

# 46. Final Research Statement

The project investigates:

> **Whether an unseen speaker's identity can be preserved while systematically manipulating prosodic characteristics of zero-shot synthesized speech, and how increasing prosodic modification affects speaker similarity, intelligibility and naturalness.**

---

# 47. Final Architecture

```text
                         USER
                          │
             ┌────────────┴────────────┐
             │                         │
      Reference Audio              Target Text
          5–10 sec                     │
             │                         │
             ▼                         ▼
     Audio Preprocessing       Text Processing
             │                         │
             ▼                         │
    Speaker Conditioning              │
             │                         │
             └────────────┬────────────┘
                          ▼
                  ZERO-SHOT TTS
                          │
                          ▼
                   BASELINE AUDIO
                          │
                          ▼
                 PROSODY CONTROLLER
                   │      │      │
                   ▼      ▼      ▼
                  F0    Energy   Rate
                   │      │      │
                   └──────┼──────┘
                          │
                          ▼
                  CONTROLLED AUDIO
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
       Speaker ID       WER      Prosody Analysis
          SECS                       F0/Energy/
                                    Duration
             │            │            │
             └────────────┼────────────┘
                          ▼
                 RESEARCH ANALYSIS
                          │
                          ▼
              IDENTITY–PROSODY TRADE-OFF
                          │
                          ▼
                  WEB APPLICATION
```

---

# 48. Golden Rule

**Do not expand the scope until the current layer works.**

Priority:

```text
ZERO-SHOT CLONING
        ↓
EVALUATION
        ↓
PITCH
        ↓
ENERGY
        ↓
RATE
        ↓
PAUSE
        ↓
RESEARCH EXPERIMENTS
        ↓
UI POLISH
        ↓
OPTIONAL EXTENSIONS
```

Never reverse this order.
