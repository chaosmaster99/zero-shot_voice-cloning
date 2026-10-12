# Models

This project does **not** train any model. All networks are pretrained, open-weight models that
are downloaded on first use (Hugging Face / SpeechBrain cache) and used for inference only:

| Role | Model | Where it is used |
|---|---|---|
| Zero-shot TTS backbone | F5-TTS `F5TTS_v1_Base` (flow matching, Vocos vocoder) | `backend/core/tts_engine.py` |
| Speaker encoder (evaluation only) | SpeechBrain `spkrec-ecapa-voxceleb` (ECAPA-TDNN) | `backend/evaluation/speaker_similarity.py` |
| ASR (WER, reference transcript) | `openai/whisper-base.en` | `backend/evaluation/asr.py` |

The *original* component of the project is the signal-processing prosody controller
(`backend/prosody/`), which needs no model weights.

Licence note: the F5-TTS checkpoint is released under CC-BY-NC; keep this in mind before any
commercial use.
