import { useMemo, useState } from 'react'

const initialSettings = {
  pitch: 0,
  energy: 0,
  rate: 1,
  pause: 1,
  emotion: 'neutral',
}

function App() {
  const [referenceFile, setReferenceFile] = useState(null)
  const [referenceText, setReferenceText] = useState('')
  const [text, setText] = useState('Welcome to our Speech Processing project.')
  const [settings, setSettings] = useState(initialSettings)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const metrics = useMemo(() => result?.metrics ?? {
    speaker_similarity: 0.0,
    wer: 0.0,
    f0_change: 0.0,
    energy_change: 0.0,
    duration: 0.0,
    rtf: 0.0,
  }, [result])

  const handleGenerate = async () => {
    if (!referenceFile) {
      setError('Please upload a reference audio file first.')
      return
    }

    setError('')
    setIsLoading(true)

    try {
      const formData = new FormData()
      formData.append('reference_audio', referenceFile)
      formData.append('reference_text', referenceText)
      formData.append('text', text)
      formData.append('pitch', String(settings.pitch))
      formData.append('energy', String(settings.energy))
      formData.append('rate', String(settings.rate))
      formData.append('pause', String(settings.pause))
      formData.append('emotion', settings.emotion)

      const response = await fetch('http://localhost:8000/synthesize', {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail || 'Synthesis failed.')
      }

      setResult(payload)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-8">
        <header className="mb-8 rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950 p-6 shadow-2xl shadow-slate-950/40">
          <p className="text-xs uppercase tracking-[0.25em] text-indigo-300">Speech Processing</p>
          <h1 className="mt-3 text-3xl font-semibold">Controllable Zero-Shot Voice Cloning</h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-300">Generate speech in a reference speaker’s voice while exploring the identity–prosody trade-off.</p>
        </header>

        <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <main className="space-y-6">
            <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-medium">Reference voice</h2>
              <div className="flex flex-wrap items-center gap-3">
                <label className="inline-flex cursor-pointer rounded-xl border border-indigo-500 bg-indigo-500/10 px-4 py-2 text-sm font-medium text-indigo-200 hover:bg-indigo-500/20">
                  Upload audio
                  <input type="file" accept="audio/*" className="hidden" onChange={(e) => setReferenceFile(e.target.files?.[0] ?? null)} />
                </label>
                <button className="rounded-xl border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-200">Record</button>
              </div>
              <div className="mt-5 h-24 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-3 text-sm text-slate-400">
                {referenceFile ? `Selected: ${referenceFile.name}` : 'No reference audio selected yet.'}
              </div>
              <input
                value={referenceText}
                onChange={(e) => setReferenceText(e.target.value)}
                className="mt-3 w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm text-slate-100 outline-none placeholder:text-slate-500"
                placeholder="Reference transcript required by F5-TTS"
              />
            </section>

            <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-medium">Target text</h2>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={5}
                className="w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm text-slate-100 outline-none ring-0 placeholder:text-slate-500"
                placeholder="Enter the text you want the speaker to say..."
              />
            </section>

            <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="mb-5 text-lg font-medium">Prosody controls</h2>
              <div className="space-y-5">
                <ControlRow label="Pitch" value={settings.pitch} min={-20} max={20} step={5} onChange={(value) => setSettings({ ...settings, pitch: value })} suffix="%" />
                <ControlRow label="Energy" value={settings.energy} min={-20} max={20} step={5} onChange={(value) => setSettings({ ...settings, energy: value })} suffix="%" />
                <ControlRow label="Speaking rate" value={settings.rate} min={0.8} max={1.4} step={0.1} onChange={(value) => setSettings({ ...settings, rate: value })} suffix="x" />
                <ControlRow label="Pause" value={settings.pause} min={0.7} max={1.4} step={0.1} onChange={(value) => setSettings({ ...settings, pause: value })} suffix="x" />
              </div>

              <div className="mt-6 flex items-center gap-3">
                <select value={settings.emotion} onChange={(e) => setSettings({ ...settings, emotion: e.target.value })} className="rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100">
                  <option value="neutral">Neutral</option>
                  <option value="excited">Excited</option>
                  <option value="calm">Calm</option>
                  <option value="sad">Sad</option>
                </select>

                <button
                  onClick={handleGenerate}
                  className="rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-500 px-5 py-2.5 text-sm font-medium text-white shadow-lg shadow-indigo-900/30 transition hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={isLoading}
                >
                  {isLoading ? 'Generating...' : 'Generate speech'}
                </button>
              </div>

              {error && <div className="mt-4 rounded-xl border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</div>}
            </section>
          </main>

          <aside className="space-y-6">
            <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-medium">Results</h2>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                <ResultCard title="Speaker similarity" value={metrics.speaker_similarity.toFixed(3)} />
                <ResultCard title="WER" value={`${(metrics.wer * 100).toFixed(1)}%`} />
                <ResultCard title="F0 delta" value={`${metrics.f0_change.toFixed(1)} Hz`} />
                <ResultCard title="Energy delta" value={metrics.energy_change.toFixed(4)} />
                <ResultCard title="Duration" value={`${metrics.duration.toFixed(2)}s`} />
                <ResultCard title="RTF" value={metrics.rtf.toFixed(3)} />
              </div>
            </section>

            <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-medium">Audio comparison</h2>
              <div className="grid gap-3">
                <AudioPanel title="Baseline" audioUrl={result?.baseline_audio} />
                <AudioPanel title="Controlled" audioUrl={result?.controlled_audio} />
              </div>
            </section>
          </aside>
        </div>
      </div>
    </div>
  )
}

function ControlRow({ label, value, min, max, step, onChange, suffix }) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
        <span>{label}</span>
        <span className="font-medium text-slate-100">{value}{suffix}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-indigo-500"
      />
    </div>
  )
}

function ResultCard({ title, value }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-3">
      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{title}</div>
      <div className="mt-2 text-xl font-semibold text-white">{value}</div>
    </div>
  )
}

function AudioPanel({ title, audioUrl }) {
  const playableUrl = audioUrl ? `http://localhost:8000/audio/${audioUrl.split(/[\\/]/).pop()}` : undefined

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-3">
      <div className="mb-2 text-sm font-medium text-slate-200">{title}</div>
      {playableUrl ? (
        <audio controls src={playableUrl} className="w-full" />
      ) : (
        <div className="h-12 rounded-lg bg-slate-900 text-xs text-slate-500 flex items-center justify-center">No audio available</div>
      )}
    </div>
  )
}

export default App
