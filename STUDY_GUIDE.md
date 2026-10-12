# Study Guide — Controllable Zero-Shot Voice Cloning with Prosodic Manipulation

How to use this guide: read Part 1 (the idea), Part 2 (theory you need), Part 3 (walk the code in the
given order), Part 4 (results), then drill Part 6 (viva questions). Exact numbers and tables are in
[`context.md`](context.md); the DSP method write-up is [`docs/PROSODY.md`](docs/PROSODY.md).

Suggested study time: ~6–8 hours (1 h idea, 2 h theory, 2–3 h code, 1 h results, 1 h viva drill).

**Current repository map.** `backend/` is the FastAPI and DSP implementation, `frontend/` is the React
studio, `experiments/` contains the reproducible study runners, `docs/results/` contains the committed
study outputs, `data/` is runtime input/output storage, and `tests/` contains fast tests that use
synthetic audio and test doubles. Model weights are not stored in the repository; Hugging Face and
SpeechBrain download them to the user cache on first use.

---

## Part 1 — The idea in 5 minutes

**What it does.** You give it 5–10 seconds of someone's voice and some text. It speaks the text in
that voice (*zero-shot voice cloning*), then lets you change **how** it is spoken — higher/lower,
more/less melodic, louder, brighter, faster, with longer/shorter pauses, or a preset like *Sad* or
*Excited*. It then **measures** whether the change really happened and what it cost.

**Two questions in one sentence:** *WHO* is speaking (identity) vs *HOW* they speak (prosody).
The project tries to change HOW while keeping WHO, and quantifies the trade-off.

**Three layers:**

1. **Pretrained neural parts** (not trained by you): F5-TTS (cloning), ECAPA (speaker similarity), Whisper (WER).
2. **Your original part**: a DSP *prosody controller* + a *measurement framework* that checks every control.
3. **Application**: FastAPI backend + React studio + experiment scripts + tests.

**The one diagram to memorise**

```
reference ─► preprocess ─┐
                         ├─► F5-TTS per sentence ─► join (0.35 s gaps) ─► BASELINE
text ─► split sentences ─┘                                                  │
                                                            ProsodyController (DSP)
                                                                            │
                                                                       CONTROLLED
        evaluate both: speaker similarity · WER · F0/energy/rate/pause/tilt → control accuracy
```

---

## Part 2 — Theory you must know

### 2.1 Speech production & prosody
- **Source–filter model**: vocal folds (source, gives F0) + vocal tract (filter, gives formants/timbre).
  Pitch/loudness/timing = *prosody*; formants/spectral envelope ≈ *identity & phonetics*.
- Prosody has four physical correlates: **F0** (pitch), **intensity** (loudness), **duration/rate/pauses**,
  **voice quality** (spectral tilt, breathiness, tremor).
- **Semitones**: perceptual pitch scale, `st = 12·log2(f/f_ref)`. ±20 % F0 ≈ ±3.2 st (+20 % → 3.16 st; −20 % → −3.86 st).
  Working in semitones makes "×range" and "+slope" additive and speaker-independent.
- **Emotion & prosody** (Murray & Arnott 1993; Scherer 2003): joy = high, wide, fast, loud, bright;
  sadness = low, narrow, slow, soft, dark, long pauses; anger = loud, bright, wide; fear = high, fast, tremulous.

### 2.2 Zero-shot TTS and F5-TTS
- *Zero-shot* = the model has never seen that speaker in training; the speaker is specified at inference
  by an audio prompt (+ its transcript).
- **F5-TTS**: non-autoregressive, **flow-matching** diffusion-style model with a Diffusion-Transformer
  backbone; text is padded to the mel length (no explicit duration model / alignment) — "fills in" the
  masked mel given the reference mel. A **Vocos** vocoder turns mel → waveform at 24 kHz.
- **NFE steps** = number of ODE solver steps; 32 default here (quality vs speed).
- Why it needs the reference **transcript**: the prompt is audio+text so the model can align the two;
  the project auto-transcribes with Whisper when missing.

### 2.3 PSOLA (the heart of the controller)
- Voiced speech is a train of pitch periods. **PSOLA** cuts pitch-synchronous windowed grains,
  then re-spaces them (changes F0) and repeats/drops grains (changes duration), overlap-adding the result.
- Formants stay roughly where they are → voice stays recognisable, which is why identity survives moderate edits.
- Praat's **Manipulation** object holds a *pitch tier* and a *duration tier*; replace them and ask for the
  overlap-add resynthesis. This is done via `parselmouth`.
- Limits: beyond ~±30 % pitch or very slow rate you hear artefacts.

### 2.4 Spectral tilt / brightness
- Vocal effort flattens the spectrum (more high-frequency energy). **Alpha ratio** =
  `10·log10(E[1–5 kHz] / E[50 Hz–1 kHz])`. A zero-phase STFT shelf raises/lowers energy above 1 kHz;
  the gain is found in closed loop until the measured alpha ratio hits the target.

### 2.5 Tremor, slope, range (intonation decomposition)
- **Range**: spread of F0 around the median (robust IQR-based), in semitones.
- **Slope**: linear rise/fall inside each phrase (declination vs lift).
- **Tremor**: periodic 5.5 Hz F0 wobble. Estimated by **least squares lock-in**: fit a slow
  piecewise-linear intonation curve + sin/cos at 5.5 Hz; amplitude = √(a²+b²).

### 2.6 Evaluation metrics
| Metric | What | Formula/notes |
|---|---|---|
| Speaker similarity | identity | cosine of ECAPA-TDNN embeddings (16 kHz) |
| WER | intelligibility | (S+D+I)/N via Levenshtein, normalised text |
| F0 median / range / slope / tremor | pitch | Praat AC tracker, octave-error fix |
| Energy | loudness | RMS over speech regions only |
| Rate | tempo | baseline speech duration / controlled speech duration |
| Pause | silence | total internal gaps ≥ 0.1 s |
| RTF | speed | TTS seconds / audio seconds |
| Control accuracy | does the knob work? | |measured − requested| |

### 2.7 Experimental design vocabulary
**Baseline**, **single-factor sweep**, **combined**, **ablation** (add controls one at a time),
**trade-off curve** (modification level 0→3 vs similarity/WER), **reproducibility metadata**
(seed, versions, GPU, git commit).

---

## Part 3 — Code walkthrough (read in this order)

| # | File | Learn |
|---|---|---|
| 1 | `backend/config.py` | every tunable and its default |
| 2 | `backend/audio/preprocessing.py` | what a "valid reference" is |
| 3 | `backend/core/text.py` | sentence splitting rules |
| 4 | `backend/core/tts_engine.py` | lazy F5-TTS load, lock, Whisper fallback transcript |
| 5 | `backend/core/pipeline.py` | `synthesize_baseline` → `synthesize_with_prosody` end-to-end |
| 6 | `backend/audio/analysis.py` | speech/phrase/pause intervals, F0, octave fix, alpha ratio, intonation decomposition |
| 7 | `backend/prosody/presets.py` | controls, limits, presets, override logic |
| 8 | `backend/prosody/controller.py` | **the novelty** — read `apply` → `_render` → `_resynthesize` → `_target_contour` → `_tilt_to` → `_scale_pauses` |
| 9 | `backend/evaluation/prosody_metrics.py`, `control_accuracy.py` | how the output is measured and compared |
| 10 | `backend/evaluation/runtime.py` + `models.py`, `speaker_similarity.py`, `asr.py`, `wer.py` | identity/intelligibility |
| 11 | `backend/api/routes/*`, `deps.py`, `schemas/` | endpoints, validation, error mapping |
| 12 | `frontend/src/App.jsx`, `components/*` | UI state: preset ↔ Custom, FormData upload |
| 13 | `experiments/run_study.py`, `run_preset_validation.py` | how results were produced |
| 14 | `tests/test_prosody.py`, `conftest.py` | how correctness is proven |

### 3.1 Request trace — `POST /synthesize` (memorise)

1. `routes/synthesis.py`: validate text (≤ 600 chars), save upload (type/size checks), call pipeline inside `guarded`.
2. `resolve_prosody(emotion, **overrides)` → preset base + overrides → validated 8-param dict.
3. `preprocess_reference_audio` → 24 kHz, normalised, trimmed, SNR warning.
4. `synthesize_baseline`: split text → F5-TTS per sentence (reference transcribed once) → trim →
   join with 0.35 s gap → peak 0.5 → `baseline.wav`.
5. `ProsodyController.from_params(params).apply(baseline)` → `controlled.wav`.
6. `evaluate_clips`: ECAPA similarity (ref vs baseline/controlled), Whisper WER, prosody metrics + contours.
7. `compute_control_accuracy(baseline_metrics, controlled_metrics, params)`.
8. Build response: URLs, metrics, accuracy, waveform/mel payloads, timings (incl. RTF), run metadata; write `_meta.json`; delete upload.

### 3.2 Controller internals — mental model

```
apply():
   remove DC
   if F0-related control active:
       measure source median F0 and range
       loop ≤ 5 times:
           render(offset, range_gain) from ORIGINAL
           measure median/range of result
           correct offset (semitones) and range_gain; stop when within tolerance
   else: render once
render():
   pauses → PSOLA(pitch tier + duration tier) → restore RMS
   → spectral tilt (closed loop on alpha ratio) → restore RMS
   → energy gain → soft limiter
```

Key formulas:
- target pitch tier: `st'(t) = offset + range·gain·st(t) + (slope/rate)·(t − t_phrase) + tremor·sin(2π·(5.5/rate)·t)`
- duration tier constant `1/rate`
- slope & tremor are given in **output time** → divide by rate in input time.
- soft limit on contour: beyond 8 st knee, `8 + 4·tanh((|x|−8)/4)`.

### 3.3 Why each tricky thing exists (bug-driven design)
| Feature | Problem it fixed |
|---|---|
| DC removal (30 Hz HPF) | Praat resynthesis subtracts mean → digital silence became noise floor → pause detection broke |
| Closed-loop median/range | open-loop missed target median by up to 13 % on real clones |
| Octave-error folding | pitch tracker halving/doubling skewed median and range |
| Level restoration after each stage | pitch/brightness otherwise leaked into loudness |
| Alpha-ratio restoration | PSOLA pitch shift changed brightness as a side effect |
| Sentence-level synthesis | F5-TTS often produced no pause to scale |
| Not-measurable pause | single sentence has nothing to stretch; 0 error would be a lie |
| Dynamics control removed | expander shifted speech/silence threshold → inaccurate & dependent on timing controls |
| Peak 0.5 baseline | headroom for +40 % energy |
| torchaudio shim | torchaudio ≥ 2.9 needs FFmpeg DLLs on Windows |

---

## Part 4 — Results you should be able to quote

- Baseline cloning: **similarity 0.813**, **WER 2.1 %** (4 voices × 5 passages).
- Control accuracy: pitch error 0.4–0.6 pp; energy ≤ 0.3 pp; rate 0.007–0.021×; pause 0.04–0.06×.
- Cost: pitch ±20 % → identity drop ≈ 0.09; rate 1.2× → 0.056; energy ±20 % ≈ 0; pauses ≈ 0.
- Trade-off: level 0→3 similarity 0.813 → 0.709 (~0.05 per level) with WER flat.
- Presets on clones: similarity 0.72–0.79, WER ≤ 3 %; mean control error tiny (pitch 0.33 pp, brightness 0.014 dB).
- Emotion recogniser barely shifts (≈ 70–80 % "neutral" everywhere) → DSP-only prosody is a weak
  emotion cue; human test needed.
- Latency: ~4–7 s warm GPU per request; RTF ≈ 0.8 in the recorded RTX 4060 study. CPU inference is
  substantially slower (the local CPU smoke run was roughly 6–7× real time), so do not quote the GPU
  number as a universal performance guarantee.

Figures to show: `docs/results/control_accuracy.png`, `identity_prosody_tradeoff.png`,
`ablation.png`, `presets.png`, `single_controls_similarity.png`.

**Interpretation sentence for slides:** *"Delivery is highly controllable and cheap for intelligibility;
the real cost is speaker identity, mostly from pitch and rate shifts."*

---

## Part 5 — Run it yourself (to understand by doing)

```powershell
# from the repository root, with .venv activated
python -m pytest -q                                           # prove the DSP and API behave
python scripts/prepare_demo_references.py                     # optional: create demo references
python -m uvicorn backend.main:app --reload --port 8000       # API and docs on :8000
cd frontend
npm install
npm run dev                                                    # studio on :5173
```

For a real web-studio run, open `http://localhost:5173`, upload or record a clear 5–10 second
reference, enter its transcript (or leave it blank for Whisper auto-transcription), enter target
text, choose a preset or adjust the controls, and select **Generate speech**. The first real request
downloads F5-TTS, Whisper, and ECAPA weights and can take several minutes on CPU.

On Windows, FFmpeg must be the **shared** build because `torchaudio`/TorchCodec needs its DLLs.
The backend automatically discovers a WinGet installation. If FFmpeg is installed elsewhere, set
`FFMPEG_BIN` in `.env` to its `bin` directory before starting Uvicorn. A quick diagnostic is:

```powershell
python -c "import backend.config, torchaudio; print(torchaudio.load('data/references/test1.wav')[0].shape)"
```

If the project is recreated on another machine, use Python 3.10–3.12 for the locked environment
when possible, install `requirements.txt`, and install the frontend dependencies with `npm install`.
Do not commit `.venv`, `node_modules`, generated audio, model weights, or experiment output folders.

Useful direct commands:

```powershell
# one zero-shot generation (reference transcript is optional)
python experiments/zero_shot/run_zero_shot.py --reference data/references/test1.wav --text "Hello there."

# DSP-only experiment on any WAV file
python experiments/evaluation/run_prosody_experiment.py --input clip.wav --output-dir out --report out/report.json

# complete multi-speaker study, after adding <speaker>.wav and <speaker>.txt files
python experiments/run_study.py --references-dir data/references --output-dir results/study
```

Exercises:
1. Change `SENTENCE_PAUSE_SECONDS` to 0.6 and see the pause-control baseline change.
2. In `presets.py` make *Happy* `pitch=+35` and watch the control-accuracy / audio artefacts.
3. Set `F5_NFE_STEPS=16`; compare RTF and WER.
4. Apply `pitch_range=2.0` on the lab and look at the F0 contour (soft limiter flattening).
5. Upload a single-sentence text and confirm pause is "not measurable".

---

## Part 6 — Viva / interview question bank

**Concept**
1. What is zero-shot voice cloning? How is the speaker specified? *(reference audio + transcript at inference; no training)*
2. What did *you* build vs reuse? *(controller, measurement/accuracy framework, pipeline, API, UI, study)*
3. Why DSP rather than the TTS model's own controls? *(course scope, interpretability, independent measurable control, WHO/HOW separation)*
4. Define prosody and name its physical correlates.
5. What is the identity–prosody trade-off and how did you quantify it? *(modification levels 0–3 vs ECAPA cosine & WER)*

**DSP**
6. Explain PSOLA. Why does it preserve formants? What breaks it?
7. Why work in semitones? Convert +20 % to semitones.
8. How is pitch range applied and why a soft limit?
9. Why generate slope/tremor in output time?
10. How is brightness controlled and why closed loop?
11. Why restore RMS after each stage? What would leak otherwise?
12. Why did you reject pYIN? How are octave errors handled?
13. How do you measure tremor? *(lock-in least squares at 5.5 Hz with slow spline)*
14. Why is pause error reported "not measurable" sometimes?
15. How does the closed loop converge and what are its tolerances? *(≤ 4 corrections; 0.05 st, 2 % range)*

**Evaluation**
16. What is ECAPA similarity? Its limitations? *(single encoder, relative indicator)*
17. How is WER computed and normalised? What if the ASR output is empty? *(1.0)*
18. What is RTF? What was yours?
19. How is control accuracy defined per control?
20. Why is the pause measured × rate?

**Results & critique**
21. Which controls cost most identity? Which are free? Why?
22. Why is the emotion recogniser unmoved? What does that say about the presets?
23. Threats to validity? *(4 voices, 1 encoder, no listening test, presets unvalidated, English only)*
24. What would you do next? *(MOS/SMOS, VCTK/LibriTTS, second encoder, word-level emphasis, breathiness/formants)*

**Engineering**
25. How do you keep the API from leaking tracebacks? *(`guarded` maps errors; request ids)*
26. How are models loaded and why lazily? How is concurrency handled? *(lazy import; `threading.Lock` on F5)*
27. How do the tests avoid needing GPUs/models? *(synthetic speech-like signal; TTS and eval test doubles)*
28. How is reproducibility ensured? *(seed, NFE, metadata.json with versions/GPU/git commit, lockfile)*
29. Ethical considerations? *(consent, impersonation notice, CC-BY-NC weights)*
30. Why can a fresh Windows setup fail before synthesis? *(TorchCodec may not find FFmpeg shared
  DLLs; the backend config adds the discovered FFmpeg `bin` directory to PATH/DLL search.)*
31. Where do model weights live? *(outside the repository in Hugging Face/SpeechBrain caches; the
  repository contains no trained or downloaded checkpoints.)*
32. Why can tests pass without downloading models? *(API and pipeline tests inject TTS/evaluation test
  doubles; a real web request still requires the pretrained weights.)*

---

## Part 7 — One-page cheat sheet

- **Models:** F5-TTS v1 Base (24 kHz, Vocos) · ECAPA-TDNN (spkrec-ecapa-voxceleb) · Whisper base.en · (SER: wav2vec2-base-superb-er).
- **Controls (8):** pitch %, range ×, slope st/s, tremor st, energy %, brightness dB, rate ×, pause ×.
- **Presets (7+custom):** neutral, happy, sad, angry, calm, excited, afraid.
- **Constants:** 24 kHz · gap 0.35 s · pause ≥ 0.1 s · speech threshold −35 dB re peak · tremor 5.5 Hz · alpha split 1 kHz · tilt shelf 900–1100 Hz · baseline peak 0.5 · ceiling 0.99 / knee 0.85 · contour knee 8 st (+4 span).
- **Headline numbers:** sim 0.813 · WER 2.1 % · −0.05 sim per level · pitch err < 0.6 pp · RTF ≈ 0.8.
- **Endpoints:** `/health /models /presets /synthesize /prosody/apply /evaluate /analyze /audio/*`.
- **Honest caveats:** 4 voices, one encoder, no human test, presets ≠ validated emotions, ±30 % artefacts.
