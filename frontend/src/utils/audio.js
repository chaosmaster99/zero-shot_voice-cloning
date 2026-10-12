// Browser-side audio helpers: decode any uploaded/recorded clip and re-encode it as 16-bit mono WAV,
// so the backend only ever has to read WAV regardless of what the browser can record.

export async function decodeAudio(blob) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext
  const context = new AudioCtx()
  try {
    const buffer = await context.decodeAudioData(await blob.arrayBuffer())
    return buffer
  } finally {
    context.close?.()
  }
}

export function encodeWav(audioBuffer) {
  const channels = audioBuffer.numberOfChannels
  const length = audioBuffer.length
  const mono = new Float32Array(length)
  for (let c = 0; c < channels; c += 1) {
    const data = audioBuffer.getChannelData(c)
    for (let i = 0; i < length; i += 1) mono[i] += data[i] / channels
  }

  const sampleRate = audioBuffer.sampleRate
  const view = new DataView(new ArrayBuffer(44 + length * 2))
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
  }
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, length * 2, true)
  for (let i = 0; i < length; i += 1) {
    const sample = Math.max(-1, Math.min(1, mono[i]))
    view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
  }
  return new Blob([view], { type: 'audio/wav' })
}

export async function toWav(blob) {
  const buffer = await decodeAudio(blob)
  return { blob: encodeWav(buffer), duration: buffer.duration }
}
