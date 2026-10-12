const PRESET_COLORS = {
  neutral: '#94a3b8',
  happy: '#fbbf24',
  sad: '#60a5fa',
  angry: '#f87171',
  calm: '#2dd4bf',
  excited: '#f472b6',
  afraid: '#c084fc',
  custom: '#a78bfa',
}

const signed = (value, digits, unit) => `${value > 0 ? '+' : ''}${Number(value).toFixed(digits)}${unit}`

const GROUPS = [
  {
    title: 'Intonation',
    controls: [
      { key: 'pitch', label: 'Pitch', hint: 'Median F0 shift', step: 1, format: (v) => signed(v, 0, '%') },
      { key: 'pitch_range', label: 'Intonation range', hint: 'Melodic variation (1.0 = natural)', step: 0.05, format: (v) => `${Number(v).toFixed(2)}×` },
      { key: 'pitch_slope', label: 'Contour slope', hint: 'Per-phrase rise (+) or fall (−)', step: 0.1, format: (v) => signed(v, 1, ' st/s') },
      { key: 'tremor', label: 'Tremor', hint: 'Shaky voice, 5.5 Hz F0 wobble', step: 0.05, format: (v) => `${Number(v).toFixed(2)} st` },
    ],
  },
  {
    title: 'Loudness & voice quality',
    controls: [
      { key: 'energy', label: 'Energy', hint: 'Speech loudness (RMS)', step: 1, format: (v) => signed(v, 0, '%') },
      { key: 'brightness', label: 'Brightness', hint: 'Vocal effort: spectral tilt above 1 kHz', step: 0.5, format: (v) => signed(v, 1, ' dB') },
    ],
  },
  {
    title: 'Timing',
    controls: [
      { key: 'rate', label: 'Speaking rate', hint: 'Speed of speech', step: 0.05, format: (v) => `${Number(v).toFixed(2)}×` },
      { key: 'pause', label: 'Pauses', hint: 'Length of pauses between phrases', step: 0.1, format: (v) => `${Number(v).toFixed(1)}×` },
    ],
  },
]

export const NEUTRAL = { pitch: 0, pitch_range: 1, pitch_slope: 0, tremor: 0, energy: 0, brightness: 0, rate: 1, pause: 1 }

export default function ProsodyPanel({ presets, limits, presetKey, params, onPreset, onParam }) {
  const active = presets.find((preset) => preset.key === presetKey)

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
      <h2 className="mb-1 text-lg font-medium">Prosody</h2>
      <p className="mb-4 text-sm text-slate-400">
        Pick an emotion preset, or choose Custom and set every control yourself. Moving a slider switches to Custom. All changes are applied
        with signal processing after synthesis.
      </p>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {presets.map((preset) => {
          const selected = preset.key === presetKey
          const color = PRESET_COLORS[preset.key] || '#94a3b8'
          return (
            <button
              key={preset.key}
              onClick={() => onPreset(preset)}
              aria-pressed={selected}
              className={`rounded-xl border px-3 py-2 text-left text-sm transition ${
                selected ? 'bg-slate-800' : 'border-slate-700 bg-slate-950/60 hover:bg-slate-800/70'
              }`}
              style={selected ? { borderColor: color } : undefined}
            >
              <span className="flex items-center gap-2 font-medium text-slate-100">
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
                {preset.label}
              </span>
            </button>
          )
        })}
      </div>
      {active && <p className="mt-3 text-sm text-slate-400">{active.description}</p>}

      <div className="mt-5 space-y-6">
        {GROUPS.map((group) => (
          <div key={group.title}>
            <h3 className="mb-3 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">{group.title}</h3>
            <div className="space-y-4">
              {group.controls.map((control) => {
                const limit = limits?.[control.key] ?? { min: 0, max: 1 }
                const value = params[control.key] ?? NEUTRAL[control.key]
                return (
                  <div key={control.key}>
                    <div className="mb-1 flex items-baseline justify-between text-sm">
                      <span className="text-slate-200">
                        {control.label} <span className="text-xs text-slate-500">{control.hint}</span>
                      </span>
                      <span className="font-medium tabular-nums text-slate-100">{control.format(value)}</span>
                    </div>
                    <input
                      type="range"
                      min={limit.min}
                      max={limit.max}
                      step={control.step}
                      value={value}
                      onChange={(event) => onParam(control.key, Number(event.target.value))}
                      className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-indigo-500"
                      aria-label={control.label}
                    />
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
