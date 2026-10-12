# context.md — Controllable Zero-Shot Voice Cloning with Prosodic Manipulation

> Single-source project context. Written to be fed to an LLM / used for PPT generation, viva
> preparation and doubt clarification. Everything below was verified against the source code and
> the committed result files in this repository. Facts are tagged by where they come from.
> A learning-oriented walkthrough lives in [`STUDY_GUIDE.md`](STUDY_GUIDE.md).

---

## 0. One-paragraph summary

A speech-processing system that **clones an unseen speaker's voice from 5–10 s of audio** using the
pretrained zero-shot TTS model **F5-TTS**, and then **controls HOW that voice speaks** with a
hand-written **DSP prosody controller** (PSOLA pitch/rate resynthesis, spectral-tilt shelving,
pause editing, energy control) exposed as eight sliders and seven emotion presets. Every control
is **measured back from the audio** (requested vs measured = *control accuracy*) and every edit's
**cost in speaker identity (ECAPA cosine) and intelligibility (Whisper WER)** is quantified. Wrapped
in a FastAPI backend and a React web studio. No model is trained or fine-tuned.

- Course: **22AIE450 – Speech Processing**, final-year semester project.
- Spec: [`PROJECT_SPEC.md`](PROJECT_SPEC.md) (1,624 lines, 48 sections).
- Core research question (RQ3): *how much prosody can be changed before speaker identity and
  intelligibility degrade?* — **WHO** is speaking is kept, **HOW** they speak is changed.

---

## 1. Problem, research questions, contribution

| Item | Content |
|---|---|
| Problem | Zero-shot TTS clones a voice but offers little explicit control of delivery. Can identity be preserved while prosody is manipulated independently? |
| RQ1 | Can a pretrained zero-shot TTS reproduce an unseen speaker from 5–10 s? |
| RQ2 | Do explicit controls (pitch, energy, rate, pauses) give predictable, measurable changes? |
| RQ3 (primary) | How does increasing prosodic manipulation affect speaker similarity, intelligibility and naturalness? |
| Contribution | NOT a new TTS architecture. A system + experimental framework: zero-shot cloning → explicit DSP prosody control → quantitative control-accuracy evaluation → identity/intelligibility cost analysis (the "identity–prosody trade-off"). |
| Out of scope (spec §6) | Training/fine-tuning TTS or speaker encoder, multilingual, word-level prosody, learned emotion model, production deployment, watermarking, multi-backbone benchmarks |

### What is original vs pretrained

| Part | Source |
|---|---|
| Zero-shot voice cloning | **Pretrained** F5-TTS `F5TTS_v1_Base` (flow-matching, Vocos vocoder), inference only |
| Speaker similarity | **Pretrained** SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`), evaluation only |
| ASR / WER, reference auto-transcript | **Pretrained** Whisper `openai/whisper-base.en` |
| Emotion check (optional experiment) | **Pretrained** `superb/wav2vec2-base-superb-er` (IEMOCAP, 4 classes), evaluation only |
| **DSP prosody controller** | **This project** — `backend/prosody/` |
| **Prosody measurement + control-accuracy framework** | **This project** — `backend/audio/analysis.py`, `backend/evaluation/` |
| Sentence-level synthesis pipeline, API, web studio, experiments, tests | **This project** |

---

## 2. System architecture

```
reference 5–10 s ─► preprocessing (mono, 24 kHz, peak-norm, trim, SNR check) ─┐
                                                                               ├─► F5-TTS (zero-shot, per sentence)
target text ─► sentence splitter ──────────────────────────────────────────────┘          │
                                                                                          ▼
                                            join sentences with 0.35 s neutral gap, peak-normalise to 0.5
                                                                                          │  = BASELINE
                                                                                          ▼
                                                                       ProsodyController (DSP, closed loop)
                                                                                          │  = CONTROLLED
                                      ┌───────────────────────────────────────────────────┤
                                      ▼                                                   ▼
        ECAPA speaker similarity · Whisper WER · F0/energy/rate/pause/alpha-ratio metrics → control accuracy
```

Key design decisions and why:

1. **DSP after TTS, not inside the model.** The project is a *speech processing* course; prosody is
   done with signal processing so WHO (TTS identity) and HOW (DSP delivery) vary independently and
   can be measured separately. (An earlier experiment with a neural expressive backbone was dropped:
   the user wanted DSP-based prosody on F5-TTS.)
2. **Sentence-level synthesis.** F5-TTS runs sentences together, leaving no pause to scale. The
   pipeline synthesizes sentence by sentence (same reference, same seed), trims edges
   (`librosa.effects.trim`, 40 dB), joins with an explicit `SENTENCE_PAUSE_SECONDS=0.35` gap. That
   gap is what the *pause* control scales.
3. **Baseline = same voice/text/backbone/seed**, so any difference to the controlled clip is purely the DSP.
4. **Baseline peak-normalised to 0.5 (−6 dBFS)** so loud presets (+40 % RMS) have headroom and the
   limiter does not flatten them.
5. **Closed-loop correction**: the controller renders, measures the output exactly as the evaluator
   does, corrects its parameters, re-renders from the original signal (never cascaded).
6. **Lazy model loading + clean errors**: evaluation models are imported on first use; a missing
   model degrades to a warning, not a crash; the API never returns a traceback.

---

## 3. Repository layout (after cleanup)

```
backend/
  main.py                    FastAPI app (CORS, lifespan cleanup of old outputs, /audio static mount)
  config.py                  env-driven settings, paths, DEFAULT_PROSODY, get_device()
  api/
    deps.py                  save_upload (type/size limits), guarded() error mapping, cleanup_old_outputs
    routes/                  health.py, synthesis.py, prosody.py, evaluation.py
    schemas/synthesis.py     Pydantic ProsodyParams, SynthesisResponse
  audio/
    preprocessing.py         read_audio, validate/preprocess_reference_audio, SNR, save_audio
    analysis.py              DC removal, speech/phrase/pause intervals, F0 extraction, octave fix,
                             alpha ratio, intonation_features (tremor/slope/range), contour_statistics
    visualization.py         waveform min/max envelope + mel image payloads for the UI
  core/
    tts_engine.py            BaseTTSEngine, F5TTSEngine (thread-locked, lazy), NullTTSEngine
    pipeline.py              synthesize_baseline, synthesize_with_prosody, apply_prosody_only
    text.py                  split_sentences (abbreviation/decimal safe, min 3 words)
  prosody/
    controller.py            ProsodyController  ← the novelty
    presets.py               PRESETS, LIMITS, resolve_prosody, validate_params
  evaluation/
    models.py                lazy loaders: ECAPA, Whisper
    speaker_similarity.py    embeddings + cosine similarity
    asr.py                   windowed Whisper transcription (30 s windows)
    wer.py                   normalize_text, Levenshtein word_error_rate
    prosody_metrics.py       analyze_prosody (all measured stats + contours)
    control_accuracy.py      requested vs measured per control
    runtime.py               evaluate_clips (ties it all together, never aborts on missing model)
  utils/
    audio_io.py              torchaudio.load → libsndfile shim (Windows/FFmpeg workaround)
    system_info.py           python/os/package versions/GPU for reproducibility
frontend/                    React 18 + Vite 5 + Tailwind 3 studio
  src/App.jsx, api.js, components/{ProsodyPanel,ReferenceInput,Results,Charts}.jsx, utils/audio.js
experiments/
  run_study.py               the full research study (32 conditions/item)
  plot_study.py              figures from results.csv
  run_preset_validation.py   7 presets on real clones (+ optional SER check)
  sentences.txt              5 fixed two-sentence passages
  zero_shot/run_zero_shot.py one cloning pass (CLI)
  evaluation/run_prosody_experiment.py   DSP-only variants + control accuracy on any WAV
  evaluation/evaluate_folder.py          similarity/WER/prosody for a folder of WAVs
scripts/download_models.py, prepare_demo_references.py
tests/                       10 test modules (signal, API, pipeline, text, presets, audio, ...)
docs/PROSODY.md, docs/results/   method write-up + committed study outputs (summary.md, results.csv, PNGs)
PROJECT_SPEC.md, IMPLEMENTATION_PLAN.md, README.md, STUDY_GUIDE.md, context.md
requirements.txt (flexible), requirements.lock.txt (exact verified env), .env.example, pytest.ini
```

---

## 4. Backend modules in detail

### 4.1 Configuration (`backend/config.py`, `.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | `auto` | `auto`/`cuda`/`cpu` (auto picks CUDA if available) |
| `MODEL_NAME` | `f5-tts` | TTS backbone key (`f5`, `f5-tts`, `f5tts` accepted) |
| `F5_MODEL` | `F5TTS_v1_Base` | checkpoint |
| `F5_NFE_STEPS` | `32` | flow-matching ODE steps (lower = faster/worse; 16 for CPU) |
| `SEED` | `42` | reproducibility |
| `SPLIT_SENTENCES` | `true` | sentence-level synthesis |
| `SENTENCE_PAUSE_SECONDS` | `0.35` | neutral inter-sentence gap |
| `SAMPLE_RATE` | `24000` | F5-TTS/Vocos native rate |
| `MIN/MAX_REFERENCE_SECONDS` | `5` / `10` | reference duration limits |
| `MAX_UPLOAD_BYTES` | 25 MB | upload limit |
| `MAX_TEXT_CHARS` | `600` | text limit |
| `ENABLE_EVALUATION` | `true` | compute similarity/WER/metrics per request |
| `CORS_ORIGINS` | `localhost:5173`, `127.0.0.1:5173` | frontend origins |
| `WHISPER_MODEL` | `openai/whisper-base.en` | ASR (read in `evaluation/models.py`) |

### 4.2 Reference preprocessing (`audio/preprocessing.py`)

`preprocess_reference_audio`: read (soundfile, mono) → validate duration 5–10 s → resample to 24 kHz →
peak-normalise to 0.95 → `librosa.effects.trim(top_db=35)` → reject if silent/too little speech
(<60 % of min duration) → crude SNR (90th vs 10th percentile frame RMS); `< 10 dB` adds a *warning*
(not an error). Returns `(audio, sr, ReferenceInfo)`.

### 4.3 TTS engine (`core/tts_engine.py`)

`F5TTSEngine.synthesize`: if no reference transcript → Whisper transcribes the reference
(`reference_text_source="whisper"`); calls `F5TTS.infer(ref_file, ref_text, gen_text, seed,
nfe_step, remove_silence=False)` under a `threading.Lock`. Model is loaded once per process.
`install_torchaudio_shim()` routes `torchaudio.load` to libsndfile because torchaudio ≥ 2.9 needs
FFmpeg shared libs on Windows.

### 4.4 Pipeline (`core/pipeline.py`)

- `synthesize_baseline`: split → per-sentence TTS → trim → join with gap + 50 ms edge pads → peak 0.5.
  Reference is transcribed **once** and reused.
- `synthesize_with_prosody`: preprocess ref → baseline → `ProsodyController.apply` → evaluate
  (reference, baseline, controlled) → control accuracy → summary + timings (reference_processing,
  tts, prosody, evaluation, total, **rtf**) + run metadata (model, seed, NFE, VRAM peak, system info).
  Writes `<id>_reference.wav`, `_baseline.wav`, `_controlled.wav`, `_meta.json`.
- `apply_prosody_only` (Prosody Lab): any 0.5–60 s recording, normalised to peak 0.5, controller,
  evaluation (WER only if `target_text` given).

### 4.5 Text (`core/text.py`)

Split after `. ! ?` when next token starts with capital/digit; keeps `Dr.`, `e.g.`, decimals;
merges fragments < 3 words into the neighbour.

### 4.6 The prosody controller (`prosody/controller.py`) — the original contribution

Eight controls:

| Control | Key | Unit | Neutral | API limits |
|---|---|---|---|---|
| Pitch | `pitch` | % change of **median F0** | 0 | −40 … +60 |
| Intonation range | `pitch_range` | × F0 variation (semitone domain) | 1.0 | 0.2 … 2.0 |
| Contour slope | `pitch_slope` | st/s added per phrase | 0 | −6 … +6 |
| Tremor | `tremor` | peak depth (st) of 5.5 Hz F0 modulation | 0 | 0 … 1.5 |
| Energy | `energy` | % change of speech RMS | 0 | −60 … +80 |
| Brightness | `brightness` | dB change of alpha ratio (spectral tilt) | 0 | −12 … +12 |
| Rate | `rate` | speaking-rate multiplier | 1.0 | 0.5 … 2.0 |
| Pause | `pause` | × internal pause length (relative to rate) | 1.0 | 0 … 3 |

Rendering chain `_render` (order matters):

1. **Pauses** (`_scale_pauses`): speech/silence split at −35 dB re. peak; any internal gap ≥ 0.1 s is
   stretched (zeros inserted at gap centre) or shrunk (centre removed); leading/trailing silence and
   short closures untouched.
2. **PSOLA resynthesis** (`_resynthesize`) — Praat *Manipulation* via `parselmouth`, one pass:
   pitch tier rewritten in semitones relative to the median:
   `st'(t) = offset + range·st(t) + slope·(t − t_phrase) + tremor·sin(2π·5.5·t)`;
   per-speaker F0 search range `0.75·Q1 … 2.5·Q3` (Hirst & De Looze); duration tier `1/rate`.
   Slope and tremor are generated in **output time** (slope/rate, 5.5/rate Hz) so a rate change doesn't
   change them. Contour is **soft-limited** beyond ±8 st (max ±12 st) to avoid creak/falsetto.
3. **Brightness** (`_tilt_to`): zero-phase STFT shelf, 0 dB < 900 Hz → g dB > 1.1 kHz; g refined in
   closed loop (≤ 3 iterations) until the alpha ratio equals input + requested; also restores the
   input's spectral balance after pitch shifting (no leakage).
4. **Closed-loop register & range** (`apply`): up to 4 corrections of `offset` (semitones) and
   `range_gain`; each time re-render from original, measure median F0 & range *as the evaluator does*.
   Reason: open-loop mapping missed the requested median by up to 13 % on real clones.
5. **Energy**: speech-region RMS restored after every stage that changes level; requested % applied
   last; `soft_limit` (tanh knee 0.85, ceiling 0.99) so RMS/energy control stays exact.

Isolation guarantee: pitch doesn't change loudness/brightness, rate doesn't change F0/slope, etc.
(`tests/test_prosody.py`). A dynamics (level-expander) control was tried and removed — it moved
speech/silence thresholds, so it could not be accurate/independent.

Other: `remove_dc` (30 Hz zero-phase high-pass) before F0 work — DC offsets corrupted pause
detection after Praat resynthesis; `is_identity` shortcut; strict validation raises `ValueError`.

### 4.7 Presets (`prosody/presets.py`)

Directions from Murray & Arnott 1993, Burkhardt & Sendlmeier 2000, Scherer 2003, Juslin & Laukka 2003.
**Starting points, not scientific emotion definitions.**

| Preset | pitch % | range | slope st/s | tremor st | energy % | bright dB | rate | pause |
|---|---|---|---|---|---|---|---|---|
| Neutral | 0 | 1.0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| Happy | +15 | 1.5 | +1.5 | 0 | +20 | +4 | 1.10 | 0.8 |
| Sad | −12 | 0.6 | −2.5 | 0.2 | −30 | −6 | 0.82 | 1.6 |
| Angry | +8 | 1.4 | −2.0 | 0 | +40 | +8 | 1.12 | 0.7 |
| Calm | −6 | 0.75 | −1.0 | 0 | −15 | −3 | 0.90 | 1.3 |
| Excited | +25 | 1.7 | +2.0 | 0 | +30 | +5 | 1.20 | 0.6 |
| Afraid | +20 | 1.2 | +0.5 | 0.6 | −10 | +2 | 1.18 | 0.9 |
| Custom | any | | | | | | | |

`resolve_prosody(emotion, **overrides)`: preset gives base values; any explicitly supplied control
(non-None) overrides it; result validated against `LIMITS`; unknown names → `ValueError`.
In the UI choosing a preset fills sliders; touching a slider switches to **Custom**.

### 4.8 Analysis & measurement (`audio/analysis.py`, `evaluation/prosody_metrics.py`)

- **F0**: Praat autocorrelation (`to_pitch_ac`), two-pass per-speaker range, `octave_jump_cost 0.6`,
  `voicing_threshold 0.5`; 10 ms step. pYIN was tried and rejected (labelled noisy frames voiced;
  moved median ±4 % under a pause-only edit).
- **Octave-error fix** (`correct_octave_errors`): frames > 7 st from running median (±0.25 s) are folded
  by whole octaves if that lands within 3 st, else dropped.
- **Tremor** = amplitude of a 5.5 Hz sinusoid fitted by least squares together with a slow
  per-phrase linear spline (knots every 0.3 s) — lock-in detection. Natural baselines ≈ 0.05–0.25 st.
- **Slope** = mean per-phrase regression slope (voiced-frame weighted) of tremor-free contour.
- **Range** (`f0_std_semitones`) = IQR/1.349 (×0.7413) of contour with tremor and slopes removed.
- **Alpha ratio** (eGeMAPS): 10·log10(E[1–5 kHz]/E[50 Hz–1 kHz]) over speech regions.
- **Speech intervals**: `librosa.effects.split(top_db=35)` relative to peak; **pauses** = internal gaps ≥ 0.1 s;
  **phrases** = speech merged across gaps < 0.1 s.
- Output dict: `f0_mean, f0_median, f0_std_semitones, f0_slope, f0_tremor, energy_mean (speech-region
  RMS), alpha_ratio_db, duration, speech_duration, rate (words/s of speech time if num_words),
  pause_count, pause_total` + optional contours (≤ 300 points).

### 4.9 Control accuracy (`evaluation/control_accuracy.py`)

Requested vs measured (and absolute error) per control:

| Control | Measured as |
|---|---|
| pitch | 100·(f0_median_ctrl / f0_median_base − 1) % |
| pitch_range | ratio of `f0_std_semitones` |
| pitch_slope | slope_after − range_ratio·rate_ratio·slope_before |
| tremor | √(tremor_after² − (range_ratio·tremor_before)²) |
| energy | 100·(RMS_ctrl / RMS_base − 1) % |
| brightness | Δ alpha ratio, dB |
| rate | speech_duration_base / speech_duration_ctrl |
| pause | (pause_total_ctrl / pause_total_base) × measured_rate; **None (not measurable)** if baseline pause < 0.05 s |

### 4.10 Identity & intelligibility

- **Speaker similarity** = cosine(ECAPA(reference), ECAPA(clip)), audio resampled to 16 kHz.
- **Identity drop** = baseline similarity − controlled similarity.
- **WER** = word-level Levenshtein vs the text the TTS was asked to say (lower-cased, punctuation
  stripped); Whisper transcript. Empty transcript = 1.0. WER delta = controlled − baseline.
- **RTF** = TTS seconds / generated audio seconds.

### 4.11 API

| Endpoint | Purpose |
|---|---|
| `GET /` | banner |
| `GET /health` | status, device, tts_available |
| `GET /models` | backbone, evaluation models, `pretrained_only: true`, system info |
| `GET /presets` | presets + slider limits |
| `POST /synthesize` | multipart: `reference_audio`, `text`, `reference_text?`, `emotion`, 8 optional controls, `seed?` → URLs + metrics + accuracy + analysis + timings |
| `POST /prosody/apply` | multipart: `audio`, `emotion`, controls, `target_text?` → controller on any recording (no TTS) |
| `POST /evaluate` | multipart: `reference_audio`, `generated_audio`, `target_text?` → similarity, WER, prosody |
| `POST /analyze` | multipart: `audio` → prosody statistics + contours |
| `GET /audio/<file>` | generated WAVs (static mount; auto-deleted after 24 h) |

Errors: `400` bad input (ValueError/FileNotFoundError/unsupported type/empty text/over 600 chars),
`413` upload > 25 MB, `503` model unavailable or GPU OOM, `500` with a request id (never a traceback).
Uploads are deleted after processing. Every response carries a synthetic-speech ethics notice.

---

## 5. Frontend (React 18 + Vite 5 + Tailwind 3)

- **Two modes**: *Voice cloning + prosody* (`/synthesize`) and *Prosody lab* (`/prosody/apply`).
- `ReferenceInput`: upload or in-browser recording (converted to WAV in `utils/audio.js`), optional transcript.
- `ProsodyPanel`: preset cards + 8 sliders in 3 groups (Intonation / Loudness & voice quality / Timing); slider → Custom.
- `Results`: reference/baseline/controlled audio players, metric tiles (similarity, identity drop, WER,
  duration, median F0, energy, RTF, total time), control-accuracy table, F0 + energy contour charts,
  waveforms, mel-spectrograms (hand-written SVG charts, no chart library).
- `api.js`: `VITE_API_URL` (default `http://localhost:8000`), friendly "backend unreachable" error.
- Dev server `http://localhost:5173`.

---

## 6. Experiments & verified results

### 6.1 Main study (`experiments/run_study.py`)

4 reference voices (`basic_ref_en`, `voice_main`, `voice_town`, `voice_country` — 3 distinct recording
conditions, bundled with F5-TTS) × 5 fixed two-sentence passages × 32 conditions = **640 evaluations**.
Machine: Windows 11, RTX 4060 Laptop 8 GB, Python 3.12.13, torch 2.11+cu128, F5-TTS 1.1.22,
NFE 32, seed 42, git commit `374779d`; ~307 s total.

Conditions: baseline; pitch ±10/±20 %; energy ±20 %; rate 0.8/1.2×; pause 0.5/2×; range 0.6/1.5×;
slope −3/+2 st/s; tremor 0.3/0.6 st; brightness ±6 dB; combined (pitch+15, energy+10, rate 1.1);
5 emotion presets; ablation A–E (baseline → +pitch → +energy → +rate → +pause+range); trade-off
sweep levels 0–3 interpolated toward STRONG_UP (pitch+20, range 1.5, energy+20, rate 1.2, pause 1.5)
and STRONG_DOWN (pitch−20, range 0.6, energy−20, rate 0.8, pause 1.5).

| Metric | Value |
|---|---|
| Zero-shot baseline speaker similarity | **0.813** (per speaker 0.75–0.87) |
| Zero-shot baseline WER | **2.1 %** |
| Pitch control error (±10/±20 %) | 0.4–0.6 percentage points |
| Energy control error (±20 %) | ≤ 0.3 pp |
| Rate control error (0.8×/1.2×) | 0.007–0.021 |
| Pause control error (0.5×/2×) | 0.04–0.06 (measurable in 20/20 items) |
| Identity drop: pitch ±20 % / rate 1.2× / energy ±20 % | 0.09 / 0.056 / ≈ 0 |
| Trade-off level 0 → 3 (up) | similarity 0.813 → 0.709, WER 2.1 % → 2.1 % |
| Warm GPU latency | ~4–7 s per multi-sentence request; TTS RTF ≈ 0.8 |

Selected rows (mean over 20 items): pitch −20 % → sim 0.723, drop 0.090; pitch −10 % → 0.767 (0.046);
pitch +10 % → 0.755 (0.059); pitch +20 % → 0.725 (0.088); rate 0.8× → 0.792 (0.021); rate 1.2× → 0.757
(0.056); pause 0.5×/2.0× → drop 0.005/0.003; range 0.6×/1.5× → drop 0.032/0.022; combined → 0.710
(0.103); ablation A→E similarity 0.813 → 0.732 → 0.731 → 0.709 → 0.724; trade-off down 0.813 → 0.776 →
0.755 → 0.712. Preset similarity in this study: happy 0.738, sad 0.769, angry 0.755, calm 0.783,
excited 0.706.

**Reading**: prosody can move far (±20 % pitch, 0.8–1.2× rate) with WER essentially unchanged; the
price is **speaker identity — roughly 0.05 similarity per modification level**, driven mostly by pitch
and rate. Energy and pauses are almost free for identity. Answers RQ2 ✔ (predictable) and RQ3 ✔ (quantified).

### 6.2 Preset validation (`experiments/run_preset_validation.py`)

4 voices × 2 sentences × 7 presets (`results/preset_validation/summary.md`):

| Control | mean abs error | max abs error |
|---|---|---|
| pitch | 0.33 pp | 2.8 pp |
| intonation range | 0.022× | 0.16× |
| contour slope | 0.58 st/s | 3.2 st/s |
| tremor | 0.14 st | 0.71 st |
| energy | 0.23 pp | 2.1 pp |
| brightness | 0.014 dB | 0.06 dB |
| rate | 0.014× | 0.06× |
| pause | 0.047× | 0.32× |

Speaker similarity 0.82 (plain clone) → 0.72–0.79 depending on preset; WER ≤ 3 %.
Independent emotion recogniser (wav2vec2 IEMOCAP, 4 class): unmodified clones ≈ 72 % "neutral"; every
preset stays 68–82 % "neutral", moving only slightly in expected directions. **Honest conclusion:**
rule-based prosody alone is a *weak* cue for perceived emotion category; breathiness, creak and
segmental detail carry much of it; a listening test is the right measure (not done).

---

## 7. Tests (`python -m pytest -q`)

| File | Covers |
|---|---|
| `test_prosody.py` | measured change == requested change for pitch/range/energy/rate/pause/slope/tremor/brightness; no leakage between controls; no clipping; stereo downmix; invalid params; DC-offset robustness; every preset reproduced within tolerance |
| `test_control_accuracy.py` | identity → zero error; combined controls; unmeasurable pause; signed error |
| `test_presets.py` | preset set, limits, directions, overrides, case-insensitivity, unknown/out-of-range rejected |
| `test_text.py` | sentence splitting, abbreviations/decimals, fragment merging |
| `test_pipeline.py` | sentence-by-sentence baseline with explicit gaps; single transcription; pause scales gaps |
| `test_api.py` | health/presets; synthesize; sad slows & lowers; custom values exact; clean 400s; no tracebacks; 503; lab; evaluate/analyze; files exist and uploads removed (uses **test-double TTS + fake eval models**) |
| `test_audio.py`, `test_audio_io.py` | reference validation (short/long/empty/corrupt/silent), preprocessing; soundfile shim contract |
| `test_evaluation.py` | WER cases |

Tests use a deterministic synthetic speech-like signal (harmonic phrases, known F0/pauses) so ground
truth is known; no model download needed.

---

## 8. Setup & run (Windows PowerShell)

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # GPU
python -m pip install -r requirements.txt            # or requirements.lock.txt for the exact env
cd frontend ; npm install ; cd ..
python scripts/download_models.py                    # optional warm-up (a few GB)
python scripts/prepare_demo_references.py            # 4 demo references into data/references
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
cd frontend ; npm run dev                            # http://localhost:5173
python -m pytest -q
python experiments/run_study.py --references-dir data/references --output-dir results/study
python experiments/run_preset_validation.py --max-sentences 2 --ser --save-audio
```

Python 3.10–3.12 recommended. API docs at `http://localhost:8000/docs`.

---

## 9. Limitations (state these honestly in a viva/PPT)

- Only 4 voices (3 recording conditions), one speaker encoder (ECAPA), no human MOS/SMOS test → numbers
  are indicative, not a benchmark.
- Post-synthesis PSOLA is natural for moderate edits; > ±30 % pitch or very slow rates sound processed.
- Prosody is utterance/phrase-global: no word-level emphasis; no learned emotion model.
- Voice quality limited to spectral tilt (no breathiness, creak, formant shift).
- Pause control needs ≥ 1 internal pause; single-sentence text → "not measurable".
- Presets are plausible starting points, not validated emotion definitions; emotion recogniser shows weak category shift.
- Speaker similarity is a relative indicator, not a verification decision.
- Only one TTS backbone (spec's 3-model benchmark reduced to F5-TTS). F5-TTS weights are **CC-BY-NC**.
- Torchaudio I/O shim needed on Windows.

## 10. Ethics

Use only recordings of consenting speakers; synthetic speech must not be used for impersonation or
deception. UI and every API response carry the notice. Non-commercial weights licence.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Zero-shot TTS | Synthesizes in an unseen speaker's voice from a short reference, no per-speaker training |
| Prosody | Melody (F0), loudness, timing, pauses, voice quality of speech |
| F0 | Fundamental frequency (pitch) of voiced speech, Hz |
| Semitone (st) | 1/12 octave; `12·log2(f/f0)` |
| PSOLA | Pitch-Synchronous Overlap-Add: change F0/duration by re-spacing and overlapping pitch-period grains |
| Praat Manipulation | Praat's PSOLA object with editable pitch and duration tiers |
| Alpha ratio | eGeMAPS spectral-tilt feature: energy 1–5 kHz over 50 Hz–1 kHz, dB |
| Tremor | Periodic F0 wobble (~4–7 Hz; here 5.5 Hz) in shaky voices |
| ECAPA-TDNN | Speaker-embedding network; cosine of embeddings = speaker similarity |
| WER | Word error rate (S+D+I)/N |
| RTF | Real-time factor = processing time / audio duration |
| NFE | Number of function evaluations (ODE steps) in flow-matching sampling |
| Control accuracy | How close the *measured* change is to the *requested* one |
| Identity drop | Baseline speaker similarity − controlled speaker similarity |
| Baseline | Zero-shot clip before DSP; same voice/text/backbone/seed |
| Closed loop | Measure output, correct parameters, re-render |

---

## 12. Quick answers (FAQ for doubts)

- **Why not use F5-TTS's own speed/emotion features?** The course is speech processing; DSP gives
  explicit, interpretable, measurable control independent of the TTS model.
- **Why does similarity drop when pitch changes?** ECAPA embeddings encode vocal-tract *and* pitch
  register; moving F0 ±20 % moves the clip away from the reference speaker's usual register.
- **Why are energy and pauses ~free for identity?** They leave spectral envelope and F0 untouched.
- **Why sentence-level synthesis?** To make pauses exist and be controllable; also a clean neutral baseline.
- **Why closed-loop?** Open-loop PSOLA maps missed the target median by up to 13 % on real voices.
- **Why peak 0.5 baseline?** Headroom for +40 % energy and bright presets.
- **Why Whisper `base.en`?** Light, English-only, good enough to compute a relative WER.
- **Is anything trained?** No. All networks are pretrained and used for inference only.
- **What is "unseen speaker"?** The speaker is defined only by the reference at inference time; none was in any training of this project.
- **Why tremor at 5.5 Hz?** Physiological vocal tremor sits at ~4–7 Hz.
- **Why is "pause" multiplied by measured rate?** So pause control is reported independently of rate.
- **Why did slope/tremor have larger errors?** Their *measurement* is noisy on short phrases (natural
  micro-modulation 0.05–0.25 st); the requested ramps are realised exactly on the same files.
- **What would improve it?** Human MOS/SMOS tests, more speakers (VCTK/LibriTTS), second speaker
  encoder, word-level emphasis, breathiness/formant control.
