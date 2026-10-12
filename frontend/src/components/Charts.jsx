import { useEffect, useRef } from 'react'

const PLOT = { width: 640, height: 150, left: 44, right: 12, top: 10, bottom: 22 }

function scale(domain, range) {
  const [d0, d1] = domain
  const [r0, r1] = range
  const span = d1 - d0 || 1
  return (value) => r0 + ((value - d0) / span) * (r1 - r0)
}

function segments(times, values, x, y) {
  const paths = []
  let current = []
  values.forEach((value, i) => {
    if (value === null || value === undefined) {
      if (current.length > 1) paths.push(current)
      current = []
    } else {
      current.push(`${x(times[i]).toFixed(1)},${y(value).toFixed(1)}`)
    }
  })
  if (current.length > 1) paths.push(current)
  return paths
}

/** Line chart for one or more contours that may contain gaps (null = unvoiced). */
export function ContourChart({ title, unit, series }) {
  const usable = series.filter((s) => s.values?.some((v) => v !== null && v !== undefined))
  if (!usable.length) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-500">
        {title}: no data
      </div>
    )
  }
  const all = usable.flatMap((s) => s.values.filter((v) => v !== null && v !== undefined))
  let low = Math.min(...all)
  let high = Math.max(...all)
  const pad = (high - low) * 0.1 || 1
  low -= pad
  high += pad
  const maxTime = Math.max(...usable.map((s) => s.times[s.times.length - 1] || 0), 0.1)
  const x = scale([0, maxTime], [PLOT.left, PLOT.width - PLOT.right])
  const y = scale([low, high], [PLOT.height - PLOT.bottom, PLOT.top])
  const ticks = [low + pad, (low + high) / 2, high - pad]

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium text-slate-200">{title}</span>
        <span className="flex gap-3 text-slate-400">
          {usable.map((s) => (
            <span key={s.label} className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </span>
      </div>
      <svg viewBox={`0 0 ${PLOT.width} ${PLOT.height}`} className="w-full" role="img" aria-label={title}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={PLOT.left} x2={PLOT.width - PLOT.right} y1={y(tick)} y2={y(tick)} stroke="#1e293b" />
            <text x={PLOT.left - 6} y={y(tick) + 3} textAnchor="end" fontSize="10" fill="#64748b">
              {tick.toFixed(0)}
            </text>
          </g>
        ))}
        <text x={PLOT.width - PLOT.right} y={PLOT.height - 6} textAnchor="end" fontSize="10" fill="#64748b">
          {maxTime.toFixed(1)} s · {unit}
        </text>
        {usable.map((s) =>
          segments(s.times, s.values, x, y).map((points, i) => (
            <polyline
              key={`${s.label}-${i}`}
              points={points.join(' ')}
              fill="none"
              stroke={s.color}
              strokeWidth="1.8"
              strokeLinejoin="round"
              opacity={s.opacity ?? 1}
            />
          )),
        )}
      </svg>
    </div>
  )
}

/** Min/max envelope waveform for several clips on a shared time axis. */
export function WaveformChart({ title, items }) {
  const usable = items.filter((item) => item.waveform?.max?.length)
  if (!usable.length) return null
  const maxDuration = Math.max(...usable.map((item) => item.duration || 0), 0.1)
  const rowHeight = 56
  const width = PLOT.width

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
      <div className="mb-1 text-xs font-medium text-slate-200">{title}</div>
      <svg viewBox={`0 0 ${width} ${rowHeight * usable.length + 14}`} className="w-full" role="img" aria-label={title}>
        {usable.map((item, row) => {
          const { min, max } = item.waveform
          const mid = row * rowHeight + rowHeight / 2
          const span = ((item.duration || maxDuration) / maxDuration) * (width - 16)
          const step = span / max.length
          const amp = rowHeight / 2 - 4
          const peak = Math.max(...max.map((v, i) => Math.max(Math.abs(v), Math.abs(min[i]))), 0.05)
          return (
            <g key={item.label}>
              <line x1="8" x2={width - 8} y1={mid} y2={mid} stroke="#1e293b" />
              {max.map((v, i) => (
                <line
                  key={i}
                  x1={8 + i * step}
                  x2={8 + i * step}
                  y1={mid - (v / peak) * amp}
                  y2={mid - (min[i] / peak) * amp}
                  stroke={item.color}
                  strokeWidth={Math.max(step, 0.6)}
                />
              ))}
              <text x={width - 8} y={row * rowHeight + 12} textAnchor="end" fontSize="10" fill="#94a3b8">
                {item.label}
              </text>
            </g>
          )
        })}
        <text x={width - 8} y={rowHeight * usable.length + 10} textAnchor="end" fontSize="10" fill="#64748b">
          {maxDuration.toFixed(1)} s
        </text>
      </svg>
    </div>
  )
}

/** Mel-spectrogram heat-map (data is [band][frame], 0..255, low frequency first). */
export function Spectrogram({ title, mel }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !mel?.data?.length) return
    const bands = mel.data.length
    const frames = mel.data[0].length
    canvas.width = frames
    canvas.height = bands
    const context = canvas.getContext('2d')
    const image = context.createImageData(frames, bands)
    for (let band = 0; band < bands; band += 1) {
      for (let frame = 0; frame < frames; frame += 1) {
        const v = mel.data[band][frame] / 255
        const row = bands - 1 - band
        const offset = (row * frames + frame) * 4
        image.data[offset] = Math.round(255 * Math.min(1, v * 2.2))
        image.data[offset + 1] = Math.round(255 * Math.max(0, Math.min(1, (v - 0.35) * 2.2)))
        image.data[offset + 2] = Math.round(255 * (0.25 + 0.75 * Math.max(0, 1 - Math.abs(v - 0.2) * 3)))
        image.data[offset + 3] = 255
      }
    }
    context.putImageData(image, 0, 0)
  }, [mel])

  if (!mel?.data?.length) return null
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
      <div className="mb-1 text-xs font-medium text-slate-200">{title}</div>
      <canvas ref={canvasRef} className="h-28 w-full rounded-lg" style={{ imageRendering: 'auto' }} />
    </div>
  )
}
