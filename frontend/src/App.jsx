import { useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function Timings({ timings }) {
  if (!timings || Object.keys(timings).length === 0) return null
  return <div className="timings">{Object.entries(timings).map(([stage, ms]) => (
    <span key={stage}>{stage.replaceAll('_', ' ')}: <strong>{ms} ms</strong></span>
  ))}</div>
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [recording, setRecording] = useState(false)
  const recorder = useRef(null)
  const chunks = useRef([])

  async function request(url, options) {
    setLoading(true); setError(''); setResult(null)
    try {
      const response = await fetch(`${API_URL}${url}`, options)
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'The request could not be completed.')
      setResult(body)
    } catch (err) {
      setError(err.message.includes('fetch') ? `Cannot reach the API at ${API_URL}. Start the backend first.` : err.message)
    } finally { setLoading(false) }
  }

  function askText(event) {
    event.preventDefault()
    if (question.trim()) request('/v1/ask/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) })
  }

  async function toggleRecording() {
    if (recording) { recorder.current.stop(); return }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      chunks.current = []
      mediaRecorder.ondataavailable = (event) => chunks.current.push(event.data)
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
        setRecording(false)
        const audio = new Blob(chunks.current, { type: mediaRecorder.mimeType || 'audio/webm' })
        const form = new FormData(); form.append('audio', audio, 'question.webm')
        request('/v1/ask/audio', { method: 'POST', body: form })
      }
      recorder.current = mediaRecorder; mediaRecorder.start(); setRecording(true)
    } catch { setError('Microphone access was denied or is unavailable.') }
  }

  return <main>
    <section className="hero"><p className="eyebrow">VOICE-ENABLED RAG</p><h1>Ask the corpus.<br />Speak naturally.</h1><p>Grounded answers from MSMARCO-XI, with citations and latency visibility.</p></section>
    <section className="card">
      <form onSubmit={askText}>
        <label htmlFor="question">Ask a question</label>
        <div className="input-row"><input id="question" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Type your question…" disabled={loading} /><button disabled={loading || !question.trim()}>Ask</button></div>
      </form>
      <div className="divider"><span>or</span></div>
      <button className={`record ${recording ? 'active' : ''}`} onClick={toggleRecording} disabled={loading}>{recording ? 'Stop & ask' : 'Record a question'}</button>
      {recording && <p className="recording">● Recording — click when you’re done</p>}
    </section>
    {loading && <p className="state">Searching retrieved context…</p>}
    {error && <p className="error">{error}</p>}
    {result && <section className={`result ${result.status}`}>
      {result.transcript && <p className="transcript">“{result.transcript}”</p>}
      <p className="status">{result.status}</p><h2>{result.answer || result.reason}</h2>
      {result.citations?.map((citation) => <article key={citation.source_id}><p>{citation.text}</p><small>Source {citation.source_id} · {citation.strategy} · score {citation.score}</small></article>)}
      <Timings timings={result.timings_ms} />
    </section>}
  </main>
}
