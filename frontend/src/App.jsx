import { useEffect, useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const LANGUAGES = [
  { code: 'ta-IN', name: 'Tamil', tag: 'TA' },
  { code: 'hi-IN', name: 'Hindi', tag: 'HI' },
  { code: 'en-IN', name: 'English', tag: 'EN' },
  { code: 'bn-IN', name: 'Bengali', tag: 'BN' },
]

const MARQUEE = 'VOICE-ENABLED RAG ◆ GROUNDED ANSWERS ◆ ENGINEERED CHUNKING ◆ GUARDRAILED ◆ <200MS PIPELINE ◆ #RAGINGOA ◆ '

const BUDGET_MS = 200

const SUGGESTIONS = [
  { lang: 'ta-IN', label: 'Tamil', prompts: [
    'இதயத் தாக்குதலின் அறிகுறிகள் என்ன?',
    'தலைவலி ஏற்படும் காரணங்கள் என்ன?',
  ] },
  { lang: 'hi-IN', label: 'Hindi', prompts: [
    'अगर कुत्ते का दौरा पड़े तो क्या करें?',
    'बुखार आने पर क्या करना चाहिए?',
  ] },
  { lang: 'en-IN', label: 'English', prompts: [
    'What should I do if my dog has a seizure?',
    'What are the symptoms of a heart attack?',
  ] },
  { lang: 'bn-IN', label: 'Bengali', prompts: [
    'কুকুরের খিঁচুটি পড়লে কী করবেন?',
    'জ্বর এলে কী করবেন?',
  ] },
]

function LatencyReport({ timings }) {
  if (!timings || Object.keys(timings).length === 0) return null
  // end_to_end_text_core is a subtotal (guardrail+retrieval+answer), not an
  // independent stage.  Exclude it from the per-stage list and compute the
  // total from the individual stages (+stt for audio queries) instead.
  const stages = Object.entries(timings).filter(([k]) => k !== 'end_to_end_text_core')
  const total = stages.reduce((a, [, v]) => a + v, 0)
  const underBudget = total <= BUDGET_MS
  const percent = Math.min((total / BUDGET_MS) * 100, 100)

  return <div className="latency">
    <div className="latency-head">
      <p className="citations-tag">Latency report</p>
      <span className={`budget-badge ${underBudget ? 'ok' : 'over'}`}>{underBudget ? '✓ Under budget' : '✕ Over budget'}</span>
    </div>
    <div className="budget-bar">
      <div className={`budget-fill ${underBudget ? 'ok' : 'over'}`} style={{ width: `${percent}%` }} />
      <span className="budget-tick" />
      <span className="budget-label">200ms budget</span>
    </div>
    <p className="latency-total"><strong>{total}ms</strong> total pipeline</p>
    <ul className="latency-stages">
      {stages.map(([stage, ms]) => (
        <li key={stage}>
          <span>{stage.replaceAll('_', ' ')}</span>
          <span className="latency-ms">{ms} ms</span>
        </li>
      ))}
    </ul>
  </div>
}

function Stat({ value, label }) {
  return <div className="stat"><strong>{value}</strong><span>{label}</span></div>
}

function LanguageSelect({ value, onChange, disabled }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  const current = LANGUAGES.find((lang) => lang.code === value)

  useEffect(() => {
    function onClickOutside(event) {
      if (ref.current && !ref.current.contains(event.target)) setOpen(false)
    }
    function onKey(event) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onClickOutside)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClickOutside)
      document.removeEventListener('keydown', onKey)
    }
  }, [])

  return <div className="lang" ref={ref}>
    <button
      type="button"
      className="lang-trigger"
      onClick={() => setOpen((v) => !v)}
      aria-haspopup="listbox"
      aria-expanded={open}
      disabled={disabled}
    >
      <span className="lang-tag">{current.tag}</span>
      <span>{current.name}</span>
      <span className="lang-caret" aria-hidden="true">{open ? '▲' : '▼'}</span>
    </button>
    {open && <ul className="lang-menu" role="listbox" aria-label="Spoken language">
      {LANGUAGES.map((lang) => <li key={lang.code}>
        <button
          type="button"
          role="option"
          aria-selected={lang.code === value}
          onClick={() => { onChange(lang.code); setOpen(false) }}
        >
          <span className="lang-tag">{lang.tag}</span>
          <span>{lang.name}</span>
          {lang.code === value && <span className="lang-check" aria-hidden="true">✓</span>}
        </button>
      </li>)}
    </ul>}
  </div>
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [recording, setRecording] = useState(false)
  const [language, setLanguage] = useState('ta-IN')
  const recorder = useRef(null)
  const chunks = useRef([])
  const canvasRef = useRef(null)
  const audioRef = useRef(null)
  const analyserRef = useRef(null)
  const rafRef = useRef(null)

  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    if (audioRef.current) { try { audioRef.current.close() } catch { /* already closed */ } }
  }, [])

  useEffect(() => {
    if (recording) startVisualizer()
    else stopVisualizer()
  }, [recording])

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
    if (question.trim()) request('/v1/ask/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, language }) })
  }

  function tryPrompt(text, lang) {
    setQuestion(text)
    setLanguage(lang)
    request('/v1/ask/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, language: lang }) })
  }

  function startVisualizer() {
    const canvas = canvasRef.current
    const analyser = analyserRef.current
    if (!canvas || !analyser) return
    const dpr = window.devicePixelRatio || 1
    const width = canvas.clientWidth || 300
    const height = 80
    canvas.width = width * dpr
    canvas.height = height * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const data = new Uint8Array(analyser.frequencyBinCount)
    const draw = () => {
      analyser.getByteFrequencyData(data)
      ctx.clearRect(0, 0, width, height)
      const bars = 48
      const step = Math.floor(data.length / bars)
      const gap = 3
      const barWidth = (width - (bars - 1) * gap) / bars
      for (let i = 0; i < bars; i++) {
        const h = (data[i * step] / 255) * (height - 4)
        ctx.fillStyle = '#FEE101'
        ctx.fillRect(i * (barWidth + gap), height - h, barWidth, h)
      }
      rafRef.current = requestAnimationFrame(draw)
    }
    draw()
  }

  function stopVisualizer() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    if (analyserRef.current) { analyserRef.current.disconnect(); analyserRef.current = null }
    const canvas = canvasRef.current
    if (canvas) { canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height) }
  }

  async function toggleRecording() {
    if (recording) { recorder.current.stop(); return }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const audioContext = new (window.AudioContext || window.webkitAudioContext)()
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      audioRef.current = audioContext
      analyserRef.current = analyser

      const mediaRecorder = new MediaRecorder(stream)
      chunks.current = []
      mediaRecorder.ondataavailable = (event) => chunks.current.push(event.data)
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
        if (audioRef.current) { try { audioRef.current.close() } catch { /* already closed */ } audioRef.current = null }
        setRecording(false)
        const audio = new Blob(chunks.current, { type: mediaRecorder.mimeType || 'audio/webm' })
        const form = new FormData(); form.append('audio', audio, 'question.webm')
        request(`/v1/ask/audio?language_code=${encodeURIComponent(language)}`, { method: 'POST', body: form })
      }
      recorder.current = mediaRecorder
      mediaRecorder.start()
      setRecording(true)
    } catch { setError('Microphone access was denied or is unavailable.') }
  }

  const totalMs = result?.timings_ms ? Object.values(result.timings_ms).reduce((a, b) => a + b, 0) : null

  return <div className="page">
    <header>
      <a className="logo" href="#top">V<span>.</span>RAG</a>
      <nav>
        <a href="#pipeline">The Pipeline</a>
        <a href="#answer">The Answer</a>
        <a href="#flow">How it Flows</a>
      </nav>
      <button className="cta" onClick={() => document.getElementById('pipeline').scrollIntoView({ behavior: 'smooth' })}>Ask</button>
    </header>

    <section className="hero" id="top">
      <p className="eyebrow">Voice-Enabled RAG · Task #2 · #RAGInGoa</p>
      <h1>Ask the corpus.<br />Speak naturally.</h1>
      <p className="hero-sub">Grounded answers from MSMARCO-XI — transcription, engineered chunking, vector retrieval and generation, wired together end to end, fast and guardrailed.</p>
      <p className="hero-meta">Tamil · Hindi · English · Bengali · &lt;200ms pipeline</p>
    </section>

    <section className="stats">
      <Stat value="4" label="Spoken languages" />
      <Stat value={totalMs !== null ? `${totalMs}ms` : '—'} label="Last pipeline latency" />
      <Stat value={result?.citations?.length ?? '—'} label="Sources cited" />
    </section>

    <div className="marquee" aria-hidden="true"><div className="marquee-inner">{MARQUEE.repeat(2)}</div></div>

    <section className="ask" id="pipeline">
      <p className="section-tag">01 — genesis day</p>
      <div className="panel">
        <label>Spoken language</label>
        <LanguageSelect value={language} onChange={setLanguage} disabled={loading} />

        <form onSubmit={askText}>
          <label htmlFor="question">Ask a question</label>
          <div className="input-row">
            <input id="question" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Type your question…" disabled={loading} />
            <button className="cta" disabled={loading || !question.trim()}>Ask</button>
          </div>
        </form>

        <div className="suggestions">
          {SUGGESTIONS.map((group) => (
            <div className="suggestion-group" key={group.lang}>
              <span className="suggestions-label">{group.label}</span>
              {group.prompts.map((p) => (
                <button type="button" className="chip" key={p} disabled={loading} onClick={() => tryPrompt(p, group.lang)}>{p}</button>
              ))}
            </div>
          ))}
        </div>

        <div className="divider"><span>or speak in the selected language</span></div>
        <button className={`record ${recording ? 'active' : ''}`} onClick={toggleRecording} disabled={loading}>{recording ? 'Stop & ask' : 'Record a question'}</button>
        {recording && <>
          <canvas ref={canvasRef} className="viz" aria-hidden="true" />
          <p className="recording">● Recording — click when you’re done</p>
        </>}
      </div>
    </section>

    {loading && <p className="state">Searching retrieved context…</p>}
    {error && <p className="error">{error}</p>}

    {result && <section className="result" id="answer">
      <p className="section-tag">02 — launch day</p>
      <div className="result-panel">
        {result.transcript && <p className="transcript">“{result.transcript}”</p>}
        <p className="status">{result.status}</p>
        <h2>{result.answer || result.reason}</h2>
        {result.citations?.length > 0 && <div className="citations">
          <p className="citations-tag">Sources</p>
          {result.citations.map((citation) => <article key={citation.source_id}>
            <p>{citation.text}</p>
            <small><span className="source-badge">Source {citation.source_id}</span> {citation.strategy} · score {citation.score}</small>
          </article>)}
        </div>}
        {result.status === 'answered' && <LatencyReport timings={result.timings_ms} />}
      </div>
    </section>}

    <section className="flow" id="flow">
      <p className="section-tag">03 — under the hood</p>
      <div className="flow-panel">
        <h2>How it flows</h2>
        <p className="flow-sub">From voice or text to a cited answer in &lt;200 ms — every stage is bounded, per-language partitioned, and measured. This is the full pipeline you just used above.</p>

        <div className="flow-diagram">
          <div className="flow-step"><span className="flow-num">01</span><strong>Capture</strong><span>Typed text or MediaRecorder audio (webm) + language tag <em>ta / hi / en / bn</em>. Empty / too-large audio is rejected before STT.</span></div>
          <div className="flow-arrow" aria-hidden="true">→</div>
          <div className="flow-step"><span className="flow-num">02</span><strong>Sarvam STT</strong><span>Isolated adapter — validates MIME, 1.2 s timeout, typed <em>STTError</em> (never silent empty transcript). Text path skips this.</span></div>
          <div className="flow-arrow" aria-hidden="true">→</div>
          <div className="flow-step"><span className="flow-num">03</span><strong>Guardrail + Language</strong><span><code>validate_question</code> blocks off-topic / injection / unsafe. <code>language_filter</code> maps <em>en-IN→en</em> etc. — try-outs set this explicitly.</span></div>
          <div className="flow-arrow" aria-hidden="true">→</div>
          <div className="flow-step"><span className="flow-num">04</span><strong>Hybrid Retrieval</strong><span>Per-language shards only: <em>lexical (BM25 sidecar, IDF≥2.0)</em> + <em>dense (hnswlib M16 ef64 or per-lang brute-force)</em> in parallel → RRF + dedup → 6 citations.</span></div>
          <div className="flow-arrow" aria-hidden="true">→</div>
          <div className="flow-step"><span className="flow-num">05</span><strong>Grounded Answer</strong><span>IDF-weighted sentence overlap (≥2 content terms, 1 distinctive). No LLM — answer is only from cited sentences, else <em>refused</em>.</span></div>
          <div className="flow-arrow" aria-hidden="true">→</div>
          <div className="flow-step"><span className="flow-num">06</span><strong>Render</strong><span>Citations + <em>timings_ms</em> (guardrail / retrieval / answer / end_to_end). Budget bar turns green &lt;200 ms. Exact cache makes repeats &lt;10 ms.</span></div>
        </div>

        <div className="flow-meta">
          <div className="flow-budget">
            <p className="citations-tag">Latency budget (warm, p50)</p>
            <ul>
              <li><span>guardrail</span><span>0.01 ms</span></li>
              <li><span>retrieval (embed+dense+lexical)</span><span>27 ms</span></li>
              <li><span>answer</span><span>0.8 ms</span></li>
              <li><strong>end-to-end core</strong><strong>28 ms</strong></li>
            </ul>
          </div>
          <div className="flow-note">
            <p className="citations-tag">Why it stays fast</p>
            <p>HNSW tuned (M16 efC128 ef64), per-language partitioning (no global scan), IDF stopword drop, query-vector cache, async lexical‖embedding, and startup warmup that pre-compiles each chip prompt. See <code>LowLatency.pdf / LowLatency2.pdf</code> in <code>backend/</code>.</p>
          </div>
        </div>
      </div>
    </section>

    <footer>
      <p><strong>© 2026 Konkan Voice RAG.</strong> All rights reserved.</p>
      <p>Built for HH Goa 2026 · Task #2 · #RAGInGoa</p>
    </footer>
  </div>
}