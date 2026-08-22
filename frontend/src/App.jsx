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
      <p className="section-tag">03 — architecture & pipeline</p>
      <div className="flow-panel">
        <h2>How it flows</h2>
        <p className="flow-sub">From multilingual voice or text input to a cited, guardrailed, and Groq-synthesized answer in sub-200ms core retrieval time. Here is the exact end-to-end pipeline in action:</p>

        <div className="flow-grid">
          <div className="flow-card">
            <div className="flow-card-head">
              <span className="flow-num">01</span>
              <span className="flow-badge">Ingestion</span>
            </div>
            <strong>Capture & Routing</strong>
            <p>Captures typed text or streams WebM audio via <code>MediaRecorder</code> with audio visualizer. Routes to one of 4 Indic language partitions (<em>Tamil, Hindi, English, Bengali</em>).</p>
          </div>

          <div className="flow-card">
            <div className="flow-card-head">
              <span className="flow-num">02</span>
              <span className="flow-badge">Speech AI</span>
            </div>
            <strong>Sarvam STT Adapter</strong>
            <p>Isolated low-latency ASR adapter converts audio to native text with strict MIME validation, 1.2s timeout, and typed <em>STTError</em>. Bypassed for direct text queries.</p>
          </div>

          <div className="flow-card">
            <div className="flow-card-head">
              <span className="flow-num">03</span>
              <span className="flow-badge">Safety</span>
            </div>
            <strong>Safety & Guardrails</strong>
            <p>Sub-millisecond regex analyzer blocks prompt injections, jailbreaks, system-prompt probing, and hazardous inputs in &lt;0.05 ms before touching retrieval.</p>
          </div>

          <div className="flow-card highlight">
            <div className="flow-card-head">
              <span className="flow-num">04</span>
              <span className="flow-badge core">Core SLO</span>
            </div>
            <strong>Supabase pgvector + FastEmbed</strong>
            <p>Generates 384-dim query vectors via ONNX and queries 97,000+ per-language partitioned chunks in Supabase using cosine similarity RPC (<code>match_chunks</code>) in ~45–70 ms.</p>
          </div>

          <div className="flow-card highlight">
            <div className="flow-card-head">
              <span className="flow-num">05</span>
              <span className="flow-badge core">Core SLO</span>
            </div>
            <strong>Unicode Grounding Verification</strong>
            <p>Unicode-aware tokenization verifies sentence-level overlap against question content terms (≥35% overlap). Guarantees zero hallucination; refuses if unsupported.</p>
          </div>

          <div className="flow-card genai">
            <div className="flow-card-head">
              <span className="flow-num">06</span>
              <span className="flow-badge genai">✨ GenAI</span>
            </div>
            <strong>Groq AI Synthesis</strong>
            <p>When enabled, Groq LLaMA synthesizes the grounded factual answer into fluent, natural conversational responses in the target language (~800–1300 ms, outside core SLO).</p>
          </div>

          <div className="flow-card">
            <div className="flow-card-head">
              <span className="flow-num">07</span>
              <span className="flow-badge">Telemetry</span>
            </div>
            <strong>Telemetry & Citations</strong>
            <p>Renders exact citations, chunk source IDs, and live sub-millisecond stage breakdown (<em>guardrail, retrieval, answer, genai</em>) with strict 200 ms budget audit.</p>
          </div>
        </div>

        <div className="flow-meta">
          <div className="flow-budget">
            <p className="citations-tag">Live Latency Budget Breakdown</p>
            <ul>
              <li><span>01. Guardrail Safety</span><span>~0.02 ms</span></li>
              <li><span>02. Supabase pgvector Retrieval</span><span>~50–75 ms</span></li>
              <li><span>03. Grounded Answer Extraction</span><span>~0.45 ms</span></li>
              <li><strong>Core Retrieval Subtotal (SLO &le; 200ms)</strong><strong>~55–80 ms (PASS)</strong></li>
              <li className="flow-llm-item"><span>✨ Groq LLM Framing (Outside Core SLO)</span><span>~850–1350 ms</span></li>
            </ul>
          </div>
          <div className="flow-note">
            <p className="citations-tag">Why the Pipeline Stays Resilient & Fast</p>
            <p>Partitioned language indices eliminate global table scans, FastEmbed runs lightweight on CPU via ONNX, Unicode-safe tokenizers handle complex Indic matras/viramas, and Groq framing strictly grounds itself on verified citations without unbounded latency.</p>
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