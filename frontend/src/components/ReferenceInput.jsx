import { useEffect, useRef, useState } from 'react'
import { toWav } from '../utils/audio'

const MAX_RECORD_SECONDS = 10

/**
 * Reference voice input: upload a file or record from the microphone.
 * Everything is normalised to a WAV blob in the browser before it is handed to the parent.
 */
export default function ReferenceInput({ value, onChange, minSeconds = 5, maxSeconds = 10, label = 'Reference voice' }) {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const recorderRef = useRef(null)
  const timerRef = useRef(null)
  const streamRef = useRef(null)

  useEffect(
    () => () => {
      clearInterval(timerRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
    },
    [],
  )

  const accept = async (blob, name) => {
    setBusy(true)
    setError('')
    try {
      const wav = await toWav(blob)
      onChange({ blob: wav.blob, name, duration: wav.duration, url: URL.createObjectURL(wav.blob) })
    } catch {
      setError('Could not decode this audio file. Try a WAV, MP3, OGG or FLAC recording.')
    } finally {
      setBusy(false)
    }
  }

  const startRecording = async () => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone recording is not supported in this browser.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const recorder = new MediaRecorder(stream)
      const chunks = []
      recorder.ondataavailable = (event) => chunks.push(event.data)
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
        clearInterval(timerRef.current)
        setRecording(false)
        accept(new Blob(chunks, { type: recorder.mimeType }), 'microphone recording')
      }
      recorderRef.current = recorder
      recorder.start()
      setRecording(true)
      setElapsed(0)
      const startedAt = Date.now()
      timerRef.current = setInterval(() => {
        const seconds = (Date.now() - startedAt) / 1000
        setElapsed(seconds)
        if (seconds >= MAX_RECORD_SECONDS && recorder.state === 'recording') recorder.stop()
      }, 100)
    } catch {
      setError('Microphone access was denied or is unavailable.')
    }
  }

  const stopRecording = () => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }

  const outOfRange = value && (value.duration < minSeconds || value.duration > maxSeconds)

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
      <h2 className="mb-1 text-lg font-medium">{label}</h2>
      <p className="mb-4 text-sm text-slate-400">
        {minSeconds}–{maxSeconds} seconds of clear speech from a speaker who has given consent.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <label className="inline-flex cursor-pointer rounded-xl border border-indigo-500 bg-indigo-500/10 px-4 py-2 text-sm font-medium text-indigo-200 hover:bg-indigo-500/20">
          Upload audio
          <input
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) accept(file, file.name)
              event.target.value = ''
            }}
          />
        </label>
        {recording ? (
          <button onClick={stopRecording} className="rounded-xl border border-red-500 bg-red-500/20 px-4 py-2 text-sm font-medium text-red-100">
            Stop ({elapsed.toFixed(1)}s)
          </button>
        ) : (
          <button onClick={startRecording} className="rounded-xl border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-200 hover:bg-slate-700">
            Record
          </button>
        )}
        {busy && <span className="text-xs text-slate-400">Decoding…</span>}
      </div>

      <div className="mt-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-3 text-sm text-slate-300">
        {value ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <span className="truncate">{value.name}</span>
              <span className={outOfRange ? 'text-amber-300' : 'text-slate-400'}>{value.duration.toFixed(1)} s</span>
            </div>
            <audio controls src={value.url} className="w-full" />
            {outOfRange && (
              <p className="text-xs text-amber-300">
                Expected {minSeconds}–{maxSeconds} s. The backend will reject clips outside this range.
              </p>
            )}
          </div>
        ) : (
          <span className="text-slate-500">No audio selected yet.</span>
        )}
      </div>
      {error && <p className="mt-3 text-sm text-red-300">{error}</p>}
    </section>
  )
}
