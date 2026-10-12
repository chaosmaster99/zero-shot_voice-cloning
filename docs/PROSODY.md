# Prosody controller — method

The prosody controller ([`backend/prosody/controller.py`](../backend/prosody/controller.py)) is the
project's original contribution. It is pure speech signal processing: it sits *after* the
zero-shot TTS backbone, so the cloned voice (speaker identity) and the delivery (prosody, voice
quality) can be varied independently and measured separately. No neural model is involved.

```
reference ──► F5-TTS (pretrained) ──► baseline ──► ProsodyController (DSP) ──► controlled
                                          └──────────── same speaker, same text, same backbone
```

## Why more than pitch and speed

A first version scaled only the median F0, the F0 range, the level, the rate and the pauses. Its
measured accuracy was good, but listeners could not *hear* an emotion: ±10 % F0 is about
±1.7 semitones, ±15 % RMS about ±1.2 dB. The acoustic literature on vocal emotion
(Murray & Arnott 1993; Burkhardt & Sendlmeier 2000; Scherer 2003; Juslin & Laukka 2003) shows
that emotions differ in more dimensions than these. The **contour shape** matters: sad speech
falls strongly, joyful speech is lifted. **Voice quality** matters: anger and joy have a flatter,
brighter spectrum (more vocal effort); sadness and calm a darker one; fear a shaky,
tremulous voice. The controller therefore has eight controls in three groups, and the presets use
much larger, literature-based values.

## Controls

| Group | Control | Meaning | Neutral | Range |
|---|---|---|---|---|
| Intonation | `pitch` | change of the **median F0**, % | 0 | −40 … +60 |
| | `pitch_range` | multiplier on the **F0 variation** (intonation depth) | 1.0 | 0.2 … 2.0 |
| | `pitch_slope` | **contour slope** added inside every phrase, semitones/second (− falls, + rises) | 0 | −6 … +6 |
| | `tremor` | depth of a **5.5 Hz F0 modulation**, semitones (shaky voice) | 0 | 0 … 1.5 |
| Loudness & voice quality | `energy` | change of **speech RMS**, % | 0 | −60 … +80 |
| | `brightness` | **spectral tilt**: change of the 1–5 kHz vs 0.05–1 kHz energy ratio, dB | 0 | −12 … +12 |
| Timing | `rate` | speaking-rate multiplier (speech *and* pauses) | 1.0 | 0.5 … 2.0 |
| | `pause` | multiplier on **internal pause length**, relative to the speech rate | 1.0 | 0 … 3 |

Units are chosen so that a request can be compared directly with a measurement
(e.g. `pitch_slope = −2.5` → the measured phrase slope falls by 2.5 st/s more than before).

## Algorithm

1. **Pauses** – speech/silence segmentation relative to the signal peak (−35 dB). Every internal
   silence of at least 100 ms is stretched (zero padding in the middle of the gap) or shrunk
   (middle of the gap removed). Leading/trailing silence and short articulatory closures are kept.
2. **Intonation and rate** – a single Praat *Manipulation* (pitch-synchronous overlap-add, PSOLA),
   so all F0 changes cost only one resynthesis:
   * F0 is tracked with a per-speaker search range (0.75·Q1 … 2.5·Q3, Hirst & De Looze);
   * the pitch tier is rewritten in semitones relative to the median *c*:
     `st'(t) = offset + range · st(t) + slope · (t − t_phrase) + tremor · sin(2π·5.5 Hz·t)`
     where `t_phrase` is the centre of the phrase (phrases = speech separated by pauses ≥ 0.1 s);
   * slope and tremor are generated in *output* time (divided by the rate), so a rate change does
     not alter them; phrases are segmented as they will be in the output (a closure that becomes
     a ≥ 0.1 s gap after slowing down is a phrase boundary there);
   * the reshaped contour is soft-limited beyond ±8 st from the median (at most ±12 st), which
     keeps wide-range voices out of the creak / falsetto regions where PSOLA sounds rough;
   * a duration tier with the constant `1/rate` sets the speaking rate.
3. **Brightness (closed loop)** – a zero-phase STFT shelf: 0 dB below 900 Hz, *g* dB above
   1.1 kHz. The gain *g* is refined from the measured alpha ratio (three iterations at most) until
   the output's alpha ratio equals the input's plus the requested change. After a pitch change it
   also restores the input's spectral balance, so pitch does not leak into brightness.
4. **Closed-loop register and range** – range expansion, spectral changes and creaky phonation
   change which frames a pitch tracker treats as voiced, so an open-loop mapping missed the
   requested median by up to 13 % on real clones. The controller therefore renders the whole chain,
   measures median F0 and range *exactly as the evaluation does*, corrects the register offset and
   the range gain, and re-renders **from the original signal** (one PSOLA generation, never
   cascaded) — at most four corrections, about 1–2 s for a 5 s clip.
5. **Energy** – after every stage that changes the level as a side effect (PSOLA, shelf) the
   speech-region RMS is restored; then the requested percentage is applied. A soft limiter (knee
   0.85, ceiling 0.99) prevents clipping. Baselines are normalised to a 0.5 peak (−6 dBFS) so that
   loud presets (+40 % RMS, brighter spectra) have headroom and are not flattened by the limiter.

Because of the level restoration and the calibrations the controls are *isolated*: changing pitch
does not change loudness or brightness, changing rate does not change F0 or the slope, and so on
(`tests/test_prosody.py`).

A dynamics control (expanding the syllable-level loudness contrast) was implemented and removed.
Any level expander pushes quiet speech edges below the speech/silence threshold. That changes the
measured speech duration, pause lengths and its own statistic, so it could not be made
accurate and independent of the timing controls.

## Emotion presets

| Preset | pitch | range | slope (st/s) | tremor (st) | energy | brightness (dB) | rate | pause |
|---|---|---|---|---|---|---|---|---|
| Neutral | 0 | 1.0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| Happy | +15 % | 1.5 | +1.5 | 0 | +20 % | +4 | 1.10 | 0.8 |
| Sad | −12 % | 0.6 | −2.5 | 0.2 | −30 % | −6 | 0.82 | 1.6 |
| Angry | +8 % | 1.4 | −2.0 | 0 | +40 % | +8 | 1.12 | 0.7 |
| Calm | −6 % | 0.75 | −1.0 | 0 | −15 % | −3 | 0.90 | 1.3 |
| Excited | +25 % | 1.7 | +2.0 | 0 | +30 % | +5 | 1.20 | 0.6 |
| Afraid | +20 % | 1.2 | +0.5 | 0.6 | −10 % | +2 | 1.18 | 0.9 |
| Custom | any | | | | | | | |

Directions follow the literature above: joy is high, wide, lifted, loud, bright and fast; sadness
is low, narrow, falling, soft, dark, slow, with long pauses and a slight waver; hot anger is
loud, very bright, wide and assertively falling; calm is lowered, even, soft and unhurried;
excitement is very high, wide, rising, loud and fast; fear is high and fast with F0 tremor.
They are starting points, **not** scientific definitions of emotion. Selecting a preset fills the
sliders; touching a slider switches to *Custom*. Over the API a preset is the base and any
explicitly supplied control overrides it.

## Sentence-level synthesis (what makes pause control meaningful)

F5-TTS tends to run sentences together, so a one-shot output often has *no* pause to scale. The
pipeline therefore synthesizes the text **sentence by sentence** (same reference, same seed),
trims each sentence's edges and joins them with an explicit neutral gap (`SENTENCE_PAUSE_SECONDS`,
default 0.35 s). That joined file is the baseline; the `pause` control then scales those gaps.
Single-sentence text has no internal pauses and the control is reported as *not measurable*.

## Measuring control accuracy

Everything is measured on speech regions only, after DC removal, by
[`backend/evaluation/prosody_metrics.py`](../backend/evaluation/prosody_metrics.py). F0 uses Praat's
autocorrelation tracker with a two-pass per-speaker range (different settings and code path from
the pulse analysis used for resynthesis). pYIN was tried first and rejected: on real speech it
labelled noisy frames as voiced and moved the median by ±4 % and the F0 spread by 1.6–2.6× under
a *pause-only* edit that cannot change F0 at all.

Octave errors (frames more than 7 st away from the running median of their ±0.25 s
neighbourhood) are folded back by whole octaves, or dropped when folding does not fit.

The voiced contour (semitones re. its median) is decomposed in one least-squares fit:

* a **slow intonation** curve per phrase (linear spline, knots every 0.3 s, bridging unvoiced gaps);
* a **5.5 Hz sinusoid** — its amplitude is the tremor (lock-in detection). Natural speech has
  almost no coherent component at that frequency, so baselines measure ≈ 0.05–0.25 st. Fitting
  the slow part per voiced run instead was tried and rejected: real voiced runs (50–100 ms) are
  shorter than one tremor cycle, and per-run lines absorbed the tremor;
* per-phrase **slopes** (regression on the tremor-free contour, weighted by voiced frames);
* the **range** = robust spread (IQR/1.349) of the contour with tremor and slopes removed, so the
  range, slope and tremor controls do not contaminate each other's measurement.

| Control | Measured as |
|---|---|
| pitch | % change of voiced median F0 |
| pitch_range | ratio of the range (above) |
| pitch_slope | slope_after − range_ratio · rate_ratio · slope_before (the natural declination is scaled by range and rate) |
| tremor | √(tremor_after² − (range_ratio · tremor_before)²) (natural micro-modulation has random phase) |
| energy | % change of speech-region RMS |
| brightness | change of the eGeMAPS **alpha ratio** of the long-term speech spectrum, dB |
| rate | baseline speech duration / controlled speech duration |
| pause | (controlled pause total / baseline pause total) × measured rate |

[`backend/evaluation/control_accuracy.py`](../backend/evaluation/control_accuracy.py) returns
requested, measured and absolute error for each control.
[`experiments/run_preset_validation.py`](../experiments/run_preset_validation.py) reproduces every
preset on real F5-TTS clones. It reports the errors, the speaker-similarity and WER cost and,
optionally, the scores of an independent pretrained speech-emotion recogniser.

## Validation on real clones

`python experiments/run_preset_validation.py --max-sentences 2 --ser` — 4 reference voices ×
2 sentences × 7 presets on F5-TTS clones (`results/preset_validation/summary.md`):

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

Slope and tremor are the noisiest measurements (short phrases; natural micro-modulation of
0.05–0.25 st), not the noisiest controls: on the same files the requested ramps and modulations
are realised exactly. Speaker similarity drops from 0.82 (plain clone) to 0.72–0.79 depending on
preset; WER stays ≤ 3 %.

**Emotion recogniser.** An independent pretrained classifier (wav2vec2-base fine-tuned on IEMOCAP,
4 classes) labels the *unmodified* clones 72 % "neutral" and keeps every preset at 68–82 %
"neutral". Its scores move only slightly in the expected directions: excited/afraid push
toward happy, calm toward happy, angry adds a little "angry". This matches the literature.
Recognition of emotions synthesised by prosody rules alone is modest, because voice quality
(breathiness, tension, creak) and segmental detail carry much of the category, and IEMOCAP-trained
recognisers generalise poorly to other speakers and recordings. A listening test is the right
measure of perceived emotion; the recogniser is reported for transparency.

## Known limitations

* PSOLA is natural for moderate edits; very large pitch shifts (beyond about ±30 %) and very
  slow rates become audibly processed.
* Voice quality is limited to spectral tilt. Breathiness, creak and formant shifts ("smiling"
  voice) are not modelled.
* Prosody is shaped per phrase and per utterance; word-level emphasis would need the text
  alignment of each word.
* Pause control needs pauses to exist: short, run-on utterances have nothing to stretch. A
  baseline without internal pauses is reported as *not measurable* instead of 0 error.
