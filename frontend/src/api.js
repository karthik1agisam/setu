const BASE = "/api";

export async function ask(question, scheme) {
  const r = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, scheme: scheme || null }),
  });
  if (!r.ok) throw new Error(`server ${r.status}`);
  return r.json();
}

export async function askAudio(blob) {
  const fd = new FormData();
  fd.append("file", blob, "query.wav");
  const r = await fetch(`${BASE}/ask/audio`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`server ${r.status}`);
  return r.json();
}

export async function fetchEvidence(chunkId) {
  const r = await fetch(`${BASE}/evidence/${encodeURIComponent(chunkId)}`);
  if (!r.ok) throw new Error(`evidence ${r.status}`);
  return r.json();
}

export async function tts(text, lang) {
  const r = await fetch(`${BASE}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, lang }),
  });
  if (!r.ok) throw new Error(`tts ${r.status}`);
  return r.blob();
}

export async function health() {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}
