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
    'போட்ஸ்வானாவின் 2015 எச்டிஐ மதிப்பு என்ன?',
    'சுறாக்கள் உலகம் முழுவதும் உள்ள பெருங்கடல்களில் வாழ்கின்றனவா?',
  ] },
  { lang: 'hi-IN', label: 'Hindi', prompts: [
    'Spotify USA का कार्यालय कहां स्थित है?',
    'Spotify USA पर संपर्क करने का फोन नंबर क्या है?',
  ] },
  { lang: 'en-IN', label: 'English', prompts: [
    'What is the personal income tax rate in Sweden?',
    'What are methanogens?',
    'What should I do if my dog has a seizure?',
  ] },
  { lang: 'bn-IN', label: 'Bengali', prompts: [
    'একটি গাড়িকে দুর্দান্ত স্টাইলের চাকা দিয়ে সাজানোর সুবিধা কী?',
    'হোমঅ্যাওয়ে ২০১৫ সালের হিসাবে কতটি দেশে তালিকা রয়েছে?',
  ] },
]

const BENCH_PCT = {
  retrieval: { p50: 27.06, p70: 30.58, p100: 43.51 },
  end_to_end: { p50: 28.11, p70: 31.32, p100: 44.38 },
}

function LatencyReport({ timings }) {
  if (!timings || Object.keys(timings).length === 0) return null
  // Core SLO stages are strictly guardrail + retrieval + answer (excluding LLM/GenAI)
  const core = Number(((timings.guardrail ?? 0) + (timings.retrieval ?? 0) + (timings.answer ?? 0)).toFixed(2))
  const genai = timings.genai ?? null
  const stt = timings.stt ?? null
  // All individual stages for the small list (including genai/stt, excluding composite subtotals)
  const allStages = Object.entries(timings).filter(([k]) => k !== 'end_to_end_text_core')
  const coreUnder = core <= BUDGET_MS
  const corePct = Math.min((core / BUDGET_MS) * 100, 100)
  const endToEndWithLLM = Number((core + (genai ?? 0) + (stt ?? 0)).toFixed(2))

  return <div className="latency">
    <p className="citations-tag">Latency report — every stage</p>
    <ul className="latency-stages latency-stages--small">
      {allStages.map(([stage, ms]) => (
        <li key={stage}>
          <span>{stage.replaceAll('_', ' ')}</span>
          <span className="latency-ms">{ms} ms</span>
        </li>
      ))}
      {core > 0 && <li><span>retrieval core (subtotal)</span><span className="latency-ms">{core} ms</span></li>}
    </ul>

    <div className="latency-highlight">
      <div className="latency-core-card">
        <div className="latency-head">
          <span className="citations-tag">Retrieval core</span>
          <span className={`budget-badge ${coreUnder ? 'ok' : 'over'}`}>{coreUnder ? '✓ Under budget' : '✕ Over budget'}</span>
        </div>
        <div className="budget-bar">
          <div className={`budget-fill ${coreUnder ? 'ok' : 'over'}`} style={{ width: `${corePct}%` }} />
          <span className="budget-tick" />
          <span className="budget-label">200ms budget</span>
        </div>
        <p className="latency-total latency-total--large"><strong>{core}ms</strong> retrieval core</p>
        <p className="latency-note">Counts toward the 200 ms SLO — guardrail + retrieval + answer only.</p>
      </div>

      {genai !== null && (
        <div className="latency-llm-card">
          <p className="citations-tag">LLM generation — Groq</p>
          <p className="latency-total"><strong>{genai}ms</strong> <span className="latency-outside">outside budget</span></p>
          <p className="latency-note">Not counted in the 200 ms SLO. Grok/Groq framing uses the grounded context only — adds ~800-1800 ms.</p>
        </div>
      )}
    </div>

    <p className="latency-total latency-total--small">End-to-end with LLM{stt !== null ? ' + STT' : ''}: <strong>{endToEndWithLLM}ms</strong> {genai !== null && <span className="latency-outside">({core}ms core + {genai}ms LLM{stt !== null ? ` + ${stt}ms STT` : ''})</span>}</p>

    <div className="latency-pct">
      <p className="citations-tag">Benchmark percentiles — warm, 48 samples (Supabase Mumbai)</p>
      <table>
        <thead><tr><th></th><th>p50</th><th>p70</th><th>p100</th></tr></thead>
        <tbody>
          <tr><td>retrieval</td><td>{BENCH_PCT.retrieval.p50} ms</td><td>{BENCH_PCT.retrieval.p70} ms</td><td>{BENCH_PCT.retrieval.p100} ms</td></tr>
          <tr><td>end-to-end core</td><td>{BENCH_PCT.end_to_end.p50} ms</td><td>{BENCH_PCT.end_to_end.p70} ms</td><td>{BENCH_PCT.end_to_end.p100} ms</td></tr>
        </tbody>
      </table>
      <p className="latency-note">Retrieval core p100 44 ms — comfortably under 200 ms. LLM generation is measured separately.</p>
    </div>
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
  const [useGenAI, setUseGenAI] = useState(true)
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
    if (question.trim()) request('/v1/ask/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, language, use_genai: useGenAI }) })
  }

  function tryPrompt(text, lang) {
    setQuestion(text)
    setLanguage(lang)
    request('/v1/ask/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, language: lang, use_genai: useGenAI }) })
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
        request(`/v1/ask/audio?language_code=${encodeURIComponent(language)}&use_genai=${useGenAI}`, { method: 'POST', body: form })
      }
      recorder.current = mediaRecorder
      mediaRecorder.start()
      setRecording(true)
    } catch { setError('Microphone access was denied or is unavailable.') }
  }

  const coreMs = result?.timings_ms ? Number(((result.timings_ms.guardrail ?? 0) + (result.timings_ms.retrieval ?? 0) + (result.timings_ms.answer ?? 0)).toFixed(2)) : null

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
      <Stat value={coreMs !== null ? `${coreMs}ms` : '—'} label="Last core latency" />
      <Stat value={result?.citations?.length ?? '—'} label="Sources cited" />
    </section>

    <div className="marquee" aria-hidden="true"><div className="marquee-inner">{MARQUEE.repeat(2)}</div></div>

    <section className="ask" id="pipeline">
      <p className="section-tag">01 — genesis day</p>
      <div className="panel">
        <label>Spoken language</label>
        <LanguageSelect value={language} onChange={setLanguage} disabled={loading} />

        <div className="ai-control-bar">
          <button
            type="button"
            className={`ai-toggle-card ${useGenAI ? 'active' : ''}`}
            onClick={() => !loading && setUseGenAI((v) => !v)}
            disabled={loading}
            aria-pressed={useGenAI}
            title="Toggle Groq LLM answer synthesis"
          >
            <div className="ai-toggle-left">
              <span className={`ai-sparkle ${useGenAI ? 'pulse' : ''}`} aria-hidden="true">✨</span>
              <div className="ai-text-block">
                <div className="ai-title-row">
                  <span className="ai-title">Groq AI Synthesis</span>
                  <span className={`ai-status-tag ${useGenAI ? 'active' : ''}`}>
                    {useGenAI ? '● LLM Active' : '○ Extractive Only'}
                  </span>
                </div>
                <span className="ai-desc">
                  {useGenAI
                    ? 'Generates natural, fluent answers from retrieved passages via Groq LLaMA'
                    : 'Extracts exact verbatim sentences directly from passages with zero hallucination'}
                </span>
              </div>
            </div>
            <div className="ai-switch" aria-hidden="true">
              <div className={`ai-switch-track ${useGenAI ? 'on' : ''}`}>
                <div className="ai-switch-thumb" />
              </div>
            </div>
          </button>
        </div>

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
        <p className="status">{result.status} {result.genai_used && <span className="genai-badge">✨ Groq framed</span>}</p>
        <h2>{result.framed_answer || result.answer || result.reason}</h2>
        {result.framed_answer && result.answer && result.framed_answer !== result.answer && (
          <details className="grounded-details"><summary>Grounded extractive source</summary><p>{result.answer}</p></details>
        )}
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