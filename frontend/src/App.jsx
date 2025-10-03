import React, { useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

const API_BASE = 'http://localhost:8000'

function useApi() {
  const upload = async (files) => {
    const form = new FormData()
    for (const f of files) form.append('pdf_files', f)
    const res = await fetch(`${API_BASE}/pdf_upload`, { method: 'POST', body: form })
    if (!res.ok) return Promise.reject(await res.text())
    return res.json()
  }
  const ask = async (query) => {
    const payload = { query, retrieval_mode: 'hybrid', max_context_chunks: 3 }
    const res = await fetch(`${API_BASE}/query_processing`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    if (!res.ok) return Promise.reject(await res.text())
    return res.json()
  }
  return { upload, ask }
}

function Human({ children }) { return <div className="msg user">{children}</div> }
function Assistant({ children }) { return <div className="msg assistant">{children}</div> }

function ShapedAnswer({ answer }) {
  if (typeof answer === 'string') {
    return <div className="markdown-content"><ReactMarkdown>{answer}</ReactMarkdown></div>
  }

  const { format_type, raw_answer } = answer

  if (format_type === 'list' && answer.items) {
    return (
      <div className="shaped-answer">
        <ul className="answer-list">
          {answer.items.map((item, idx) => (
            <li key={idx}><ReactMarkdown>{item}</ReactMarkdown></li>
          ))}
        </ul>
      </div>
    )
  }

  if (format_type === 'steps' && answer.steps) {
    return (
      <div className="shaped-answer">
        <ol className="answer-steps">
          {answer.steps.map((step, idx) => (
            <li key={idx}>
              <strong>Step {step.num}:</strong> <ReactMarkdown>{step.text}</ReactMarkdown>
            </li>
          ))}
        </ol>
      </div>
    )
  }

  if (format_type === 'definition') {
    return (
      <div className="shaped-answer">
        <div className="answer-definition">
          <p><strong>Definition:</strong> <ReactMarkdown>{answer.definition}</ReactMarkdown></p>
          {answer.details && answer.details.length > 0 && (
            <div className="answer-details">
              {answer.details.map((detail, idx) => (
                <div key={idx}><ReactMarkdown>{detail}</ReactMarkdown></div>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  if (format_type === 'general' && answer.paragraphs) {
    return (
      <div className="shaped-answer">
        {answer.paragraphs.map((para, idx) => (
          <div key={idx}><ReactMarkdown>{para}</ReactMarkdown></div>
        ))}
      </div>
    )
  }

  return <div className="markdown-content"><ReactMarkdown>{raw_answer || JSON.stringify(answer)}</ReactMarkdown></div>
}

export default function App(){
  const { upload, ask } = useApi()
  const [ready, setReady] = useState(false)
  const [phase, setPhase] = useState('idle') // idle | uploading | asking
  const [chat, setChat] = useState([])
  const [perFile, setPerFile] = useState(null)
  const inputRef = useRef()
  const fileRef = useRef()
  const [error, setError] = useState('')

  const themeHeader = useMemo(() => (
    <div className="hero">
      <h1 style={{ color: '#111' }}>
        Stack AI <span style={{ color: 'var(--accent)' }}>RAG</span> Pipeline
      </h1>
    </div>
  ), [])

  const onUpload = async (e) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    setPhase('uploading'); setError('')
    try{
      const r = await upload(files)
      setPerFile(prev => ({ ...prev, ...r.per_file_chunks }))
      setReady(true)
      if (r.skipped_files?.length) {
        setError(`${r.message || 'Some files were skipped'}`)
      }
    }catch(err){
      setError(String(err))
    }finally{ setPhase('idle') }
  }

  const onAsk = async () => {
    const q = inputRef.current.value.trim()
    if (!q) return
    setChat(prev => [...prev, { role: 'user', content: q }])
    inputRef.current.value = ''
    setPhase('asking'); setError('')
    try{
      const r = await ask(q)
      const sources = (r.sources || []).map((s, idx) => {
        const name = s?.metadata?.filename || `source-${idx+1}`
        const text = s?.text || ''
        return { name, snippet: text.slice(0, 80).replace(/\s+/g,' ') }
      })
      setChat(prev => [...prev, { role:'assistant', content: r.answer, sources }])
    }catch(err){
      setError(String(err))
    }finally{ setPhase('idle') }
  }

  return (
    <div className="container">
      {themeHeader}

      <div className="card" style={{marginTop: 8}}>
        <div className="col">
          <div className="row" style={{justifyContent:'space-between'}}>
            <div className="col">
              <label className="muted" htmlFor="pdf-upload">Upload PDFs</label>
              <input id="pdf-upload" aria-label="Upload PDFs" ref={fileRef} className="file-hidden" type="file" accept="application/pdf" multiple onChange={onUpload} />
              <button className="red" disabled={phase==='uploading'} onClick={()=>fileRef.current && fileRef.current.click()}>
                {phase==='uploading' ? 'Preparing Knowledge Base...' : 'Select PDFs'}
              </button>
            </div>
            <div className="chips">
              <span className="chip">Retrieval: Hybrid</span>
            </div>
          </div>
          {perFile && Object.keys(perFile).length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div className="muted" style={{ marginBottom: 6 }}>Knowledge Base ({Object.keys(perFile).length} files):</div>
              <div className="file-list">
                {Object.entries(perFile).map(([filename, chunks]) => (
                  <div key={filename} className="file-item">
                    <span className="file-name">{filename}</span>
                    <span className="chunk-count">{chunks} chunks</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{marginTop: 16}}>
        <div className="row">
          <input ref={inputRef} type="text" placeholder="Ask a question about your PDFs..." onKeyDown={(e)=>{if(e.key==='Enter') onAsk()}} />
          <button onClick={onAsk} disabled={!ready || phase!=='idle'}>{phase==='asking' ? 'Retrieving answer...' : 'Ask'}</button>
          <button className="secondary" onClick={()=>setChat([])}>Clear</button>
        </div>
        {!ready && <div className="muted" style={{marginTop:8}}>Upload PDFs first.</div>}
        {error && <div className="muted" style={{color:'var(--accent)', marginTop:8}}>{error}</div>}
        <div className="chat">
          {chat.map((m,i)=> (
            <div key={i}>
              {m.role==='user' ? <Human>{m.content}</Human> : <Assistant>
                <ShapedAnswer answer={m.content} />
                {!!m.sources?.length && (
                  <div className="sources">
                    {m.sources.map((s, j)=> (
                      <span key={j} className="source">{s.name} · {s.snippet}</span>
                    ))}
                  </div>
                )}
              </Assistant>}
            </div>
          ))}
        </div>
      </div>

      <div className="footer">RAG Agent · hybrid retrieval · cites PDF sources</div>
    </div>
  )
}


