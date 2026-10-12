export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

async function request(path, options) {
  let response
  try {
    response = await fetch(`${API_URL}${path}`, options)
  } catch {
    throw new Error(`Cannot reach the backend at ${API_URL}. Is it running?`)
  }
  let payload = null
  try {
    payload = await response.json()
  } catch {
    /* non-JSON error body */
  }
  if (!response.ok) {
    const detail = payload?.detail
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}).`)
  }
  return payload
}

export const getPresets = () => request('/presets')
export const getHealth = () => request('/health')

export const synthesize = (formData) => request('/synthesize', { method: 'POST', body: formData })
export const applyProsody = (formData) => request('/prosody/apply', { method: 'POST', body: formData })

export const audioUrl = (path) => (path ? `${API_URL}${path}` : undefined)
