import { useEffect, useMemo, useRef, useState } from 'react'

const REFERENCE_PROMPT = 'Hello, this is my voice sample for the speech processing project. I am speaking clearly and naturally so my voice can be cloned.'
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
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
  const [isRecording, setIsRecording] = useState(false)
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const [recordingPreviewUrl, setRecordingPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const mediaRecorderRef = useRef(null)
  const recordingChunksRef = useRef([])
  const recordingTimerRef = useRef(null)

  const metrics = useMemo(() => result?.metrics ?? {
    speaker_similarity: 0.0,
    wer: 0.0,
    f0_change: 0.0,
    energy_change: 0.0,
    duration: 0.0,
    rtf: 0.0,
  }, [result])

  useEffect(() => () => {
    if (recordingTimerRef.current) {
      window.clearInterval(recordingTimerRef.current)
    }
    if (recordingPreviewUrl) {
      URL.revokeObjectURL(recordingPreviewUrl)
    }
  }, [recordingPreviewUrl])

  const handleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop()
      return
    }

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('Audio recording is not supported in this browser. Try the latest Chrome or Edge.')
      return
    }

    try {
      setError('')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg']
        .find((type) => MediaRecorder.isTypeSupported(type))
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)

      recordingChunksRef.current = []
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data)
        }
      }
      recorder.onerror = () => {
        stream.getTracks().forEach((track) => track.stop())
        setIsRecording(false)
        setError('The recording could not be completed. Please check microphone permissions and try again.')
      }
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop())
        if (recordingTimerRef.current) {
          window.clearInterval(recordingTimerRef.current)
          recordingTimerRef.current = null
        }

        try {
          const recordedBlob = new Blob(recordingChunksRef.current, { type: recorder.mimeType || 'audio/webm' })
          const wavBlob = await convertToWav(recordedBlob)
          const file = new File([wavBlob], 'recorded-reference.wav', { type: 'audio/wav' })
          setReferenceFile(file)
          setReferenceText(REFERENCE_PROMPT)
          setRecordingPreviewUrl(URL.createObjectURL(wavBlob))
          setError('')
        } catch {
          setError('The recording format could not be converted. Please try recording again in Chrome or Edge.')
        } finally {
          setIsRecording(false)
        }
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setRecordingSeconds(0)
      setIsRecording(true)
      recordingTimerRef.current = window.setInterval(() => {
        setRecordingSeconds((seconds) => seconds + 1)
      }, 1000)
    } catch {
      setError('Microphone access was denied or unavailable. Allow microphone access and try again.')
    }
  }

  const handleGenerate = async () => {
    if (!referenceFile) {
      setError('Please upload a reference audio file first.')
      return
    }
    if (!referenceText.trim()) {
      setError('Please enter the exact sentence spoken in the reference audio.')
      return
    }
    if (!text.trim()) {
      setError('Please enter text to synthesize.')
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

      const controller = new AbortController()
      const timeoutId = window.setTimeout(() => controller.abort(), 10 * 60 * 1000)
      let response
      try {
        response = await fetch(`${API_BASE_URL}/synthesize`, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        })
      } finally {
        window.clearTimeout(timeoutId)
      }

      const responseText = await response.text()
      let payload
      try {
        payload = JSON.parse(responseText)
      } catch {
        throw new Error(`The backend returned an invalid response (HTTP ${response.status}).`)
      }
      if (!response.ok) {
        throw new Error(payload.detail || 'Synthesis failed.')
      }

      setResult(payload)
    } catch (err) {
      setError(err.name === 'AbortError' ? 'Synthesis timed out after 10 minutes. Check the backend logs and try again.' : err.message)
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
                  <input
                    type="file"
                    accept="audio/*"
                    className="hidden"
                    onChange={(e) => {
                      setReferenceFile(e.target.files?.[0] ?? null)
                      setReferenceText(REFERENCE_PROMPT)
                      setError('')
                    }}
                  />
                </label>
                <button
                  type="button"
                  onClick={handleRecording}
                  className={`rounded-xl border px-4 py-2 text-sm text-slate-200 ${isRecording ? 'border-red-500 bg-red-500/15 text-red-200' : 'border-slate-700 bg-slate-800 hover:bg-slate-700'}`}
                >
                  {isRecording ? `Stop recording (${recordingSeconds}s)` : 'Record sample'}
                </button>
              </div>
              <div className="mt-4 rounded-xl border border-indigo-500/30 bg-indigo-500/10 p-3 text-sm text-indigo-100">
                <div className="font-medium">Read this sentence aloud (5–10 seconds)</div>
                <p className="mt-1 text-indigo-200">{REFERENCE_PROMPT}</p>
                <p className="mt-2 text-xs text-indigo-300">Your recording will be converted to WAV and used as the reference voice.</p>
              </div>
              <div className="mt-5 h-24 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-3 text-sm text-slate-400">
                {referenceFile ? (
                  <div>
                    <div>Selected: {referenceFile.name}</div>
                    {recordingPreviewUrl && <audio controls src={recordingPreviewUrl} className="mt-2 w-full" />}
                  </div>
                ) : 'No reference audio selected yet.'}
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

async function convertToWav(blob) {
  const audioContext = new AudioContext()
  try {
    const audioBuffer = await audioContext.decodeAudioData(await blob.arrayBuffer())
    const channelCount = audioBuffer.numberOfChannels
    const sampleCount = audioBuffer.length
    const wavBuffer = new ArrayBuffer(44 + sampleCount * 2)
    const view = new DataView(wavBuffer)
    const writeString = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)))
    const writeUint16 = (offset, value) => view.setUint16(offset, value, true)
    const writeUint32 = (offset, value) => view.setUint32(offset, value, true)

    writeString(0, 'RIFF')
    writeUint32(4, 36 + sampleCount * 2)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    writeUint32(16, 16)
    writeUint16(20, 1)
    writeUint16(22, 1)
    writeUint32(24, audioBuffer.sampleRate)
    writeUint32(28, audioBuffer.sampleRate * 2)
    writeUint16(32, 2)
    writeUint16(34, 16)
    writeString(36, 'data')
    writeUint32(40, sampleCount * 2)

    const channels = Array.from({ length: channelCount }, (_, index) => audioBuffer.getChannelData(index))
    for (let index = 0; index < sampleCount; index += 1) {
      const sample = channels.reduce((sum, channel) => sum + channel[index], 0) / channelCount
      const clamped = Math.max(-1, Math.min(1, sample))
      view.setInt16(44 + index * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
    }

    return new Blob([wavBuffer], { type: 'audio/wav' })
  } finally {
    await audioContext.close()
  }
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
  const playableUrl = audioUrl ? `${API_BASE_URL}/audio/${audioUrl.split(/[\\/]/).pop()}` : undefined

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
