import { audioUrl } from '../api'
import { ContourChart, Spectrogram, WaveformChart } from './Charts'

const COLORS = { reference: '#94a3b8', baseline: '#38bdf8', controlled: '#f472b6' }

function fmt(value, digits = 2, suffix = '') {
  return value === null || value === undefined || Number.isNaN(value) ? '—' : `${Number(value).toFixed(digits)}${suffix}`
}

function Metric({ title, value, sub }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
      <div className="text-[11px] uppercase tracking-[0.18em] text-slate-400">{title}</div>
      <div className="mt-1 text-xl font-semibold text-white">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

function AudioCard({ title, src, color }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
        <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
        {title}
      </div>
      {src ? <audio controls src={src} className="w-full" /> : <div className="text-xs text-slate-500">No audio</div>}
    </div>
  )
}

const CONTROL_LABELS = {
  pitch: 'Pitch (median F0)',
  pitch_range: 'Intonation range',
  pitch_slope: 'Contour slope (added)',
  tremor: 'Tremor depth (added)',
  energy: 'Energy (RMS)',
  brightness: 'Brightness (alpha ratio)',
  rate: 'Speaking rate',
  pause: 'Pauses',
}

function ControlTable({ accuracy }) {
  const rows = Object.entries(accuracy || {})
  if (!rows.length) return null
  return (
    <div className="overflow-hidden rounded-xl border border-slate-800">
      <table className="w-full text-sm">
        <thead className="bg-slate-900 text-xs uppercase tracking-wider text-slate-400">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Control</th>
            <th className="px-3 py-2 text-right font-medium">Requested</th>
            <th className="px-3 py-2 text-right font-medium">Measured</th>
            <th className="px-3 py-2 text-right font-medium">Error</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800 bg-slate-950/60">
          {rows.map(([key, row]) => {
            const digits = row.unit === '%' || row.unit === 'dB' ? 1 : 2
            const unit = row.unit === 'x' || row.unit === '%' ? row.unit : ` ${row.unit}`
            return (
              <tr key={key}>
                <td className="px-3 py-2 text-slate-200">{CONTROL_LABELS[key] || key}</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmt(row.requested, digits, unit)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmt(row.measured, digits, unit)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-400">{fmt(row.abs_error, digits, row.unit === '%' ? ' pp' : row.unit === 'x' ? '' : unit)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default function Results({ result, mode }) {
  if (!result) return null
  const { metrics = {}, urls = {}, analysis = {}, evaluation = {}, timings = {}, reference } = result
  const clips = evaluation.clips || {}
  const refProsody = evaluation.reference?.prosody
  const baselineLabel = mode === 'lab' ? 'Original' : 'Baseline (no prosody)'
  const hasReference = Boolean(urls.reference)

  const f0Series = [
    ...(hasReference && refProsody?.contours
      ? [{ label: 'Reference', color: COLORS.reference, times: refProsody.contours.times, values: refProsody.contours.f0, opacity: 0.55 }]
      : []),
    { label: baselineLabel, color: COLORS.baseline, times: clips.baseline?.prosody?.contours?.times ?? [], values: clips.baseline?.prosody?.contours?.f0 ?? [] },
    { label: 'Controlled', color: COLORS.controlled, times: clips.controlled?.prosody?.contours?.times ?? [], values: clips.controlled?.prosody?.contours?.f0 ?? [] },
  ]
  const energySeries = [
    { label: baselineLabel, color: COLORS.baseline, times: clips.baseline?.prosody?.contours?.energy_times ?? [], values: clips.baseline?.prosody?.contours?.energy_db ?? [] },
    { label: 'Controlled', color: COLORS.controlled, times: clips.controlled?.prosody?.contours?.energy_times ?? [], values: clips.controlled?.prosody?.contours?.energy_db ?? [] },
  ]
  const waveItems = [
    ...(hasReference ? [{ label: 'Reference', color: COLORS.reference, ...analysis.reference }] : []),
    { label: baselineLabel, color: COLORS.baseline, ...analysis.baseline },
    { label: 'Controlled', color: COLORS.controlled, ...analysis.controlled },
  ]

  return (
    <section className="space-y-6">
      {(result.warnings || []).length > 0 && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-3">
        {hasReference && <AudioCard title="Reference (preprocessed)" src={audioUrl(urls.reference)} color={COLORS.reference} />}
        <AudioCard title={baselineLabel} src={audioUrl(urls.baseline)} color={COLORS.baseline} />
        <AudioCard title={`Controlled · ${result.emotion}`} src={audioUrl(urls.controlled)} color={COLORS.controlled} />
      </div>

      {reference?.transcript && (
        <p className="text-xs text-slate-500">
          Reference transcript ({reference.transcript_source === 'whisper' ? 'auto-transcribed' : 'provided'}): “{reference.transcript}”
        </p>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          title="Speaker similarity"
          value={fmt(metrics.speaker_similarity, 3)}
          sub={mode === 'lab' ? 'original vs controlled (ECAPA cosine)' : `baseline ${fmt(metrics.baseline_speaker_similarity, 3)} · ECAPA cosine`}
        />
        <Metric
          title="Identity drop"
          value={fmt(metrics.identity_drop, 3)}
          sub="similarity lost to prosody control"
        />
        <Metric
          title="WER"
          value={metrics.wer === null || metrics.wer === undefined ? '—' : `${(metrics.wer * 100).toFixed(1)}%`}
          sub={metrics.baseline_wer === null || metrics.baseline_wer === undefined ? 'Whisper ASR' : `baseline ${(metrics.baseline_wer * 100).toFixed(1)}%`}
        />
        <Metric
          title="Duration"
          value={fmt(metrics.duration, 2, ' s')}
          sub={`baseline ${fmt(metrics.baseline_duration, 2, ' s')}`}
        />
        <Metric title="Median F0" value={fmt(metrics.f0_median, 0, ' Hz')} sub={`baseline ${fmt(metrics.baseline_f0_median, 0, ' Hz')}`} />
        <Metric title="Speech energy" value={fmt(metrics.energy, 3)} sub={`baseline ${fmt(metrics.baseline_energy, 3)}`} />
        {timings.rtf !== undefined && <Metric title="Real-time factor" value={fmt(timings.rtf, 2)} sub={`TTS ${fmt(timings.tts, 1, ' s')}`} />}
        {timings.total !== undefined && <Metric title="Total time" value={fmt(timings.total, 1, ' s')} sub={`prosody ${fmt(timings.prosody, 2, ' s')}`} />}
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium uppercase tracking-wider text-slate-400">Control accuracy</h3>
        <ControlTable accuracy={result.control_accuracy} />
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium uppercase tracking-wider text-slate-400">Speech analysis</h3>
        <div className="grid gap-3 lg:grid-cols-2">
          <ContourChart title="F0 contour" unit="Hz" series={f0Series} />
          <ContourChart title="Energy contour" unit="dB" series={energySeries} />
          <WaveformChart title="Waveform" items={waveItems} />
          <div className="grid gap-3">
            <Spectrogram title={`Mel-spectrogram · ${baselineLabel}`} mel={analysis.baseline?.mel} />
            <Spectrogram title="Mel-spectrogram · Controlled" mel={analysis.controlled?.mel} />
          </div>
        </div>
      </div>

      {result.notice && <p className="text-xs text-slate-500">{result.notice}</p>}
    </section>
  )
}
