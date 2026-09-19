import { useEffect, useRef, useState } from "react";
import { ask, askAudio, fetchEvidence, health, tts } from "./api.js";

const VERDICT_STYLE = {
  eligible: { label: "ELIGIBLE", cls: "v-eligible" },
  ineligible: { label: "NOT ELIGIBLE", cls: "v-ineligible" },
  cannot_determine: { label: "CANNOT DETERMINE", cls: "v-unknown" },
  unknown: { label: "UNKNOWN", cls: "v-unknown" },
};

export default function App() {
  const [lang, setLang] = useState("en");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recording, setRecording] = useState(false);
  const [evidence, setEvidence] = useState({});
  const [backend, setBackend] = useState(null);
  const rec = useRef(null);

  useEffect(() => {
    health().then(setBackend).catch(() => setBackend(null));
  }, []);

  async function submit(q) {
    setLoading(true);
    setError(null);
    setResult(null);
    setEvidence({});
    try {
      const r = q ? await ask(q) : null;
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function toggleMic() {
    if (recording) {
      rec.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      const chunks = [];
      mr.ondataavailable = (e) => chunks.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        setLoading(true);
        setError(null);
        setResult(null);
        try {
          const r = await askAudio(new Blob(chunks, { type: "audio/webm" }));
          setResult(r);
        } catch (e) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      };
      rec.current = mr;
      mr.start();
      setRecording(true);
    } catch {
      setError("microphone unavailable");
    }
  }

  async function showEvidence(cid) {
    if (evidence[cid]) return;
    try {
      const c = await fetchEvidence(cid);
      setEvidence((m) => ({ ...m, [cid]: c }));
    } catch {
      /* card stays collapsed */
    }
  }

  async function speak() {
    if (!result?.answer) return;
    try {
      const blob = await tts(result.answer, result.detected_language || lang);
      new Audio(URL.createObjectURL(blob)).play();
    } catch {
      setError("tts failed");
    }
  }

  const v = result && (VERDICT_STYLE[result.verdict] || VERDICT_STYLE.unknown);

  return (
    <div className="page">
      <header>
        <h1>SETU</h1>
        <p className="tag">
          Verified welfare-scheme eligibility — every claim cited to official evidence
        </p>
        <div className="status">
          {backend
            ? `backend ok · ${backend.chunks} evidence chunks · mt: ${backend.translator_backend}`
            : "backend offline"}
        </div>
      </header>

      <div className="query-bar">
        <select value={lang} onChange={(e) => setLang(e.target.value)}>
          <option value="en">English</option>
          <option value="hi">हिन्दी</option>
          <option value="te">తెలుగు</option>
        </select>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && question && submit(question)}
          placeholder="Ask about a welfare scheme…"
        />
        <button onClick={() => submit(question)} disabled={!question || loading}>
          Ask
        </button>
        <button
          className={recording ? "mic rec" : "mic"}
          onClick={toggleMic}
          title={recording ? "Stop" : "Voice query"}
        >
          {recording ? "■" : "🎙"}
        </button>
      </div>

      {loading && <p className="note">Verifying against official evidence… (up to ~40s)</p>}
      {error && <p className="error">Error: {error}</p>}

      {result && (
        <div className="result">
          <div className="verdict-row">
            <span className={`verdict ${v.cls}`}>{v.label}</span>
            <span className={`gate gate-${result.gate}`}>
              {result.gate === "release" ? "verified" : result.gate}
            </span>
            {result.scheme && <span className="chip">{result.scheme}</span>}
            <span className="chip">{result.detected_language}</span>
          </div>

          {result.transcript && (
            <p className="note">Heard: “{result.transcript}”</p>
          )}

          {result.answer ? (
            <div className="answer">
              <p>{result.answer}</p>
              <button className="mini" onClick={speak}>▶ listen</button>
            </div>
          ) : (
            <div className="answer withheld">
              <p>
                {result.gate === "clarify"
                  ? "Need more information to determine eligibility:"
                  : `Answer withheld — ${result.gate_reason || "insufficient verified evidence"}`}
              </p>
              {result.missing?.length > 0 && (
                <ul>
                  {result.missing.map((m) => <li key={m}>{m}</li>)}
                </ul>
              )}
            </div>
          )}

          {result.claim_unsupported_rate > 0 && (
            <p className="note">
              unsupported-claim rate: {result.claim_unsupported_rate.toFixed(2)}
            </p>
          )}

          {result.assessments?.length > 0 && (
            <section>
              <h3>Conditions checked</h3>
              <ul className="conds">
                {result.assessments.map((a, i) => (
                  <li key={i} className={`c-${a.status}`}>
                    <b>{a.status}</b> — {a.description}
                    {a.user_fact && <em> (you: {a.user_fact})</em>}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {result.evidence?.length > 0 && (
            <section>
              <h3>Evidence (official sources)</h3>
              {result.evidence.map((e) => (
                <div key={e.chunk_id} className="ev-card">
                  <div className="ev-head" onClick={() => showEvidence(e.chunk_id)}>
                    <span className="mono">{e.chunk_id}</span>
                    <span>p.{e.page} · {e.section_path}</span>
                    <a href={e.source_url} target="_blank" rel="noreferrer">source ↗</a>
                  </div>
                  {evidence[e.chunk_id] && (
                    <p className="ev-text">{evidence[e.chunk_id].text}</p>
                  )}
                </div>
              ))}
            </section>
          )}

          {result.timings_ms?.total_ms && (
            <p className="note">pipeline: {(result.timings_ms.total_ms / 1000).toFixed(1)}s</p>
          )}
        </div>
      )}

      <footer>
        SETU never generates eligibility rules — it cites official documents or abstains.
      </footer>
    </div>
  );
}
