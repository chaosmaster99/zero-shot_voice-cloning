import { useEffect, useMemo, useState } from 'react'
import { applyProsody, getHealth, getPresets, synthesize } from './api'
import ProsodyPanel, { NEUTRAL } from './components/ProsodyPanel'
import ReferenceInput from './components/ReferenceInput'
import Results from './components/Results'

const FALLBACK_PRESETS = [{ key: 'neutral', label: 'Neutral', description: 'No manipulation.', params: NEUTRAL }]
const FALLBACK_LIMITS = {
  pitch: { min: -40, max: 60 },
  pitch_range: { min: 0.2, max: 2 },
  pitch_slope: { min: -6, max: 6 },
  tremor: { min: 0, max: 1.5 },
  energy: { min: -60, max: 80 },
  brightness: { min: -12, max: 12 },
  rate: { min: 0.5, max: 2 },
  pause: { min: 0, max: 3 },
}

function App() {
  const [mode, setMode] = useState('clone') // clone | lab
  const [presets, setPresets] = useState(FALLBACK_PRESETS)
  const [limits, setLimits] = useState(FALLBACK_LIMITS)
  const [health, setHealth] = useState(null)
  const [reference, setReference] = useState(null)
  const [referenceText, setReferenceText] = useState('')
  const [text, setText] = useState('Welcome to our speech processing project. Today we explore how prosody shapes meaning.')
  const [presetKey, setPresetKey] = useState('neutral')
  const [params, setParams] = useState(NEUTRAL)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [resultMode, setResultMode] = useState('clone')
  const [error, setError] = useState('')

  useEffect(() => {
    getPresets()
      .then((payload) => {
        setPresets(payload.presets)
        setLimits(payload.limits)
      })
      .catch((err) => setError(err.message))
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  const choosePreset = (preset) => {
    setPresetKey(preset.key)
    if (preset.key !== 'custom') setParams({ ...NEUTRAL, ...preset.params })
  }

  const changeParam = (key, value) => {
    setPresetKey('custom')
    setParams((current) => ({ ...current, [key]: value }))
  }

  const canGenerate = useMemo(() => {
    if (loading || !reference) return false
    return mode === 'lab' || text.trim().length > 0
  }, [loading, reference, mode, text])

  const generate = async () => {
    setError('')
    setLoading(true)
    try {
      const form = new FormData()
      form.append(mode === 'lab' ? 'audio' : 'reference_audio', reference.blob, 'reference.wav')
      form.append('emotion', presetKey)
      Object.entries(params).forEach(([key, value]) => form.append(key, String(value)))
      let payload
      if (mode === 'lab') {
        form.append('target_text', referenceText)
        payload = await applyProsody(form)
      } else {
        form.append('text', text)
        form.append('reference_text', referenceText)
        payload = await synthesize(form)
      }
      setResult(payload)
      setResultMode(mode)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-8">
        <header className="mb-6 rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950 p-6 shadow-2xl shadow-slate-950/40">
          <p className="text-xs uppercase tracking-[0.25em] text-indigo-300">Speech Processing · 22AIE450</p>
          <h1 className="mt-3 text-3xl font-semibold">Controllable Zero-Shot Voice Cloning</h1>
          <p className="mt-2 max-w-3xl text-sm text-slate-300">
            Clone an unseen voice from a few seconds of speech, then reshape <em>how</em> it speaks with signal processing — pitch, intonation
            range and contour, tremor, loudness, voice brightness, rate and pauses — while measuring what it costs in speaker identity and
            intelligibility.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
            <span className={`rounded-full border px-3 py-1 ${health?.tts_available ? 'border-emerald-500/50 text-emerald-300' : 'border-amber-500/50 text-amber-300'}`}>
              {health ? (health.tts_available ? `F5-TTS ready · ${health.device}` : 'TTS unavailable — Prosody Lab still works') : 'Backend offline'}
            </span>
          </div>
        </header>

        <div className="mb-6 inline-flex rounded-xl border border-slate-800 bg-slate-900 p-1 text-sm">
          {[
            ['clone', 'Voice cloning + prosody'],
            ['lab', 'Prosody lab (any recording)'],
          ].map(([key, label]) => (
            <button
              key={key}
              onClick={() => setMode(key)}
              className={`rounded-lg px-4 py-2 transition ${mode === key ? 'bg-indigo-500 text-white' : 'text-slate-300 hover:text-white'}`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="grid gap-6 xl:grid-cols-2">
          <div className="space-y-6">
            <ReferenceInput
              value={reference}
              onChange={setReference}
              label={mode === 'lab' ? 'Speech recording' : 'Reference voice'}
              minSeconds={mode === 'lab' ? 0.5 : 5}
              maxSeconds={mode === 'lab' ? 60 : 10}
            />

            <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="mb-3 text-lg font-medium">{mode === 'lab' ? 'Transcript (optional)' : 'Text'}</h2>
              <label className="mb-1 block text-xs text-slate-400">
                {mode === 'lab' ? 'Transcript of the recording — enables WER' : 'Transcript of the reference — leave blank to auto-transcribe with Whisper'}
              </label>
              <input
                value={referenceText}
                onChange={(event) => setReferenceText(event.target.value)}
                className="mb-4 w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm outline-none placeholder:text-slate-600"
                placeholder={mode === 'lab' ? 'What is said in the recording' : 'What is said in the reference recording'}
              />
              {mode === 'clone' && (
                <>
                  <label className="mb-1 block text-xs text-slate-400">Text to speak in the cloned voice</label>
                  <textarea
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    rows={4}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm outline-none placeholder:text-slate-600"
                    placeholder="Enter the text you want the cloned voice to say…"
                  />
                </>
              )}
            </section>
          </div>

          <div className="space-y-6">
            <ProsodyPanel presets={presets} limits={limits} presetKey={presetKey} params={params} onPreset={choosePreset} onParam={changeParam} />

            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={generate}
                disabled={!canGenerate}
                className="rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-500 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-900/30 transition hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? 'Working…' : mode === 'lab' ? 'Apply prosody' : 'Generate speech'}
              </button>
              {!reference && <span className="text-sm text-slate-400">Add a recording first.</span>}
              {loading && mode === 'clone' && <span className="text-sm text-slate-400">First run downloads/loads the models — this can take a minute.</span>}
            </div>
            {error && <div className="rounded-xl border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}
          </div>
        </div>

        {result && (
          <div className="mt-10 border-t border-slate-800 pt-8">
            <h2 className="mb-5 text-xl font-semibold">Results</h2>
            <Results result={result} mode={resultMode} />
          </div>
        )}

        <footer className="mt-12 border-t border-slate-800 pt-4 text-xs text-slate-500">
          Generated speech is synthetic. Only use voices you have permission to clone, and never use this system for impersonation or deception.
        </footer>
      </div>
    </div>
  )
}

export default App
