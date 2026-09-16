# Flagship AI Project Deep-Research Report
### Two Level-D flagship projects for an exceptionally strong 4th-year CS student

**Prepared:** September 2026
**Standard applied:** Think like an elite research engineer. Build like a very strong 4th-year student. 12–20 weeks, one person, ~$0–50 budget, public data only.

---

## LANDSCAPE OVERVIEW (concise)

Three structural facts shape what's worth building in 2026:

1. **"LLM + RAG + React" is the new tutorial.** Every candidate has one. What differentiates now is *evaluation rigor* (golden sets, groundedness gates, failure taxonomies), *real constraints* (offline, edge, low-resource language, latency budgets), and *real users*.
2. **The biggest open problem in GenAI isn't generation — it's trust.** Groundedness verification, eval harnesses, and honest refusal are where production systems actually break and where a student can show real depth.
3. **In DL, image classification is dead as a signal.** What still signals depth: segmentation, unsupervised anomaly detection, domain shift/generalization, weak supervision, temporal modeling, edge deployment, and uncertainty quantification.

The two selected flagships exploit gaps that are real, documented, and under-served:

- **GenAI:** Verified, multilingual welfare-scheme eligibility navigator (India) — a documented exclusion problem where hallucination literally causes harm.
- **DL:** Multisensor satellite flood-extent mapping with weak supervision and cross-event generalization — a real disaster-response problem with a public benchmark that leaves large headroom.

---

## SECTION 1 — REAL-WORLD PROBLEM LANDSCAPE

### Problems investigated and triaged

| Domain | Problem | Real? | Data obtainable? | Student-feasible? | Verdict |
|---|---|---|---|---|---|
| Public services | Citizens miss welfare schemes due to awareness/navigability | Yes — ~38% of PM-JAY-eligible households unaware (see below) | Yes — myScheme, scheme guidelines PDFs | Yes | **FLAGSHIP (GenAI)** |
| Disaster response | Rapid flood-extent mapping for relief | Yes — floods displace millions/yr; Cloud to Street built a company on this | Yes — Sen1Floods11 public | Yes, on free GPUs | **FLAGSHIP (DL)** |
| Accessibility | Deaf/HoH communication without interpreters | Yes — ~5M+ ISL users in India | Yes — INCLUDE dataset | Yes | Finalist |
| Developer ops | On-call incident triage/postmortems | Yes | Partially — public postmortems exist but telemetry is hard | Medium | Finalist (GenAI) |
| Low-resource NLP | Telugu/Indian-language voice services | Yes — 90M+ Telugu speakers, weak tooling | Yes — IndicVoices, Vistaar, CommonVoice | Yes | Finalist (GenAI) |
| Manufacturing | Visual anomaly detection | Yes | Yes — MVTec AD, VisA | Yes | Serious candidate |
| Healthcare | DR/glaucoma screening | Yes | Yes — APTOS, EyePACS, REFUGE | Yes | Serious but crowded |
| Legal/MSME | Contract review for freelancers | Yes | Partial — few public Indian contracts | Medium | Watchlist |
| Education | Misconception-aware tutoring | Yes | Yes — NCERT public | Medium | Watchlist |
| Agriculture | Crop disease severity (not just class) | Yes | Yes — PlantVillage | Yes | Serious but crowded |
| Urban infra | Pothole/road-quality mapping | Yes — Indian cities | Partial — must self-collect | Medium | Watchlist |
| Conservation | Bioacoustic poaching/species detection | Yes | Partial — fragmented datasets | Medium | Watchlist |
| Finance | Fraud detection | Yes | Yes | Yes | **Rejected — generic** |
| Prediction tasks | Stock/house prices | — | — | — | **Rejected — tutorial** |

### Why the welfare-scheme problem is real (evidence)

- A peer-reviewed six-state study of PM-JAY (India's national health insurance) found **only ~62% of eligible households were aware of the scheme**, and awareness of *own eligibility* was ~78% among the aware — with the most marginalized least aware. (PubMed 36478057)
- Dvara Research's *State of Exclusion* report documents that awareness campaigns alone fail; citizens face high "search costs" navigating enrolment procedures across schemes with regional variation.
- myScheme (myscheme.gov.in) aggregates thousands of central/state schemes but is a keyword-search portal in formal English — it does not answer "am I eligible?" as a reasoning problem over eligibility clauses.
- Eligibility is a **multi-constraint logical problem** (income AND caste category AND district AND age AND occupation…). A wrong "yes" sends a poor family through a paperwork dead-end; a wrong "no" costs them benefits. This is precisely where LLM hallucination is dangerous — making the *groundedness gate* the technically interesting core.

### Why flood mapping is real (evidence)

- Sen1Floods11 (Bonafilia et al., CVPRW 2020; Cloud to Street): 4,831 labeled 512×512 chips of Sentinel-1 SAR + Sentinel-2 optical across 11 global flood events and 6 continents. Published benchmark mIoU ≈ 53% (Remote Sensing 2021 follow-up) — *enormous headroom* and a built-in evaluation story.
- Only ~446 chips are hand-labeled; the rest are weak labels — **the paper itself identifies weak supervision as the bottleneck**, which is exactly the interesting student-scale research angle (cross-modal distillation literature exists to build on: Environmental Data Science 2023).
- SAR sees through clouds — critical for monsoon India, where optical imagery is useless exactly when floods happen. Real operational relevance (NDMA, ISRO/NRSC, disaster NGOs).

---

## SECTION 2 — GENERATIVE AI CANDIDATES (15)

| # | Idea | Problem | Differentiation lever | Risk |
|---|---|---|---|---|
| G1 | **Verified scheme navigator (multilingual)** | Citizens can't determine scheme eligibility | Claim-level groundedness gate + eligibility-reasoning eval set + Telugu/Hindi | Content ops (keeping docs current) |
| G2 | On-call incident co-pilot | Postmortems take hours; evidence scattered | Ground postmortem in metrics/logs; eval on public postmortem corpus | Telemetry data is simulated |
| G3 | Repo-grounded code review agent | LLM review hallucinates context | AST/diff grounding, verified-comment-only output | Crowded (Copilot, CodeRabbit) |
| G4 | Voice-first Telugu public-service agent | Low-literacy users can't use text portals | IndicWhisper fine-tune + constrained dialog + TTS | ASR quality on dialects |
| G5 | Agent observability + eval harness | Agents silently degrade in prod | "CI for agent behavior": trace → score → gate | Needs an agent to wrap; crowded OSS |
| G6 | Legal contract assistant for MSMEs | Freelancers sign bad contracts | Clause-level retrieval over Indian Contract Act + risk taxonomy | Thin public data |
| G7 | Court-judgment GraphRAG | Case law research is expensive | Entity/statute graph over public judgments | Huge corpus; indexing cost |
| G8 | Vernacular form-filling doc intelligence | Govt forms in English exclude users | Vision + constrained generation to structured schema | Data collection needed |
| G9 | Research-paper replication agent | Reproducing papers is painful | Paper→runnable-code agent, eval vs Papers-with-Code | Very hard eval; scope creep |
| G10 | SQL/BI agent with self-verification | Small biz owners can't query their data | Query→execute→verify→repair loop | Text-to-SQL is well-trodden |
| G11 | NCERT-grounded tutoring w/ misconception diagnosis | Generic tutors don't diagnose *why* | Student-model + Socratic policy + syllabus grounding | Pedagogy eval is fuzzy |
| G12 | Grievance/RTI drafting agent | Citizens can't draft effective complaints | Template-grounded drafting + jurisdiction routing | Narrow but real |
| G13 | Medical-report explainer | Patients can't read lab reports | Guideline-grounded explanations + hard "see a doctor" gates | Safety/regulatory optics |
| G14 | Domain SLM via distillation | API LLMs too costly/latent for narrow tasks | Distill a 0.5–1B model for one task; latency/cost study | Narrow demo |
| G15 | Meeting→field-report agent (construction/site) | Site engineers lose hours writing reports | Domain-term ASR + structured extraction | Niche audience |

### GenAI audit summary
- G3, G5, G10 are most vulnerable to "a company/OSS already did this."
- G9 is intellectually sexy but eval-prohibitively hard for a student.
- **G1 stands alone**: real harmed population, hard technical core (grounded multi-constraint reasoning), India relevance, public data, and a natural multilingual extension that most global solutions can't touch.

---

## SECTION 3 — DEEP LEARNING CANDIDATES (15)

| # | Idea | Task | Data | Differentiation lever | Risk |
|---|---|---|---|---|---|
| D1 | **Satellite flood mapping (SAR+optical)** | Semantic segmentation | Sen1Floods11 (4,831 chips, public) | Weak supervision + cross-event generalization + fusion ablation | Geospatial tooling curve |
| D2 | ISL word-level recognition → edge | Temporal classification | INCLUDE (4,287 videos, 263 signs) | Real-time on-device, signer-independence, domain shift lab→webcam | "Gesture recognition" can read as common |
| D3 | Industrial anomaly detection | Unsupervised seg | MVTec AD / VisA | Uncertainty + edge + few-shot | Academically crowded |
| D4 | DR screening on smartphone | Classification | APTOS/EyePACS/RFMiD | Domain-shift + on-device + per-class sensitivity | Very common student domain |
| D5 | Bioacoustic threat detection | Audio classification | ESC-50 + fragmented field data | Lab→field domain shift | Data assembly painful |
| D6 | Crop disease *severity* grading | Ordinal regression/seg | PlantVillage + augmentation | Severity (not class) is rarer + actionable | Still agriculture-crowded |
| D7 | Structural crack segmentation | Segmentation | SDNET2018 | Infrastructure inspection angle | Moderate |
| D8 | CCTV traffic anomaly detection | Video anomaly | DoTA/CADP | Real-time + weak labels | Heavy compute |
| D9 | PM2.5/air-quality nowcasting | Spatiotemporal forecast | OpenAQ/CPCB open data | Graph+temporal; India cities | Station coverage sparse |
| D10 | Camera-trap species ID | Classification w/ domain shift | iWildCam, SpeciesNet | Location-generalization is a real published problem | Large dataset |
| D11 | Pothole/road-quality mapping | Detection + severity | Self-collected + public | Real Indian urban problem | Self-collection burden |
| D12 | Underwater debris detection | Detection | Trash-ICRA19 etc. | Environmental angle | Niche |
| D13 | Badminton/sports shot analysis | Video understanding | Self-annotation | Fun demo | Self-labeling burden |
| D14 | PCB defect inspection | Detection | DeepPCB | Manufacturing relevance | Small dataset, less depth |
| D15 | Water-level estimation from CCTV | Regression from video | Scarce public data | Novel | Data problem is fatal |

### DL audit summary
- D4/D6 are real but the "train CNN on Kaggle medical/agri dataset" genre is the single most common student DL project — only viable with a *very* strong second act (edge + domain shift).
- D2 is demo-gold (live webcam) but isolated-sign recognition is close to solved at benchmark level (94.5% on INCLUDE-50) — the differentiation must come from *deployment constraints and generalization*, not accuracy.
- **D1 stands alone**: segmentation not classification, multisensor fusion, *weak supervision is the published bottleneck*, cross-event domain generalization is a real research question, SAR data is genuinely exotic on a student resume, and there's a live operational story (you can map a real recent flood).

---

## SECTION 4 — EXISTING-SOLUTION & COMPETITION ANALYSIS

### G1 — Scheme navigator
- **Existing:** myScheme portal (keyword search, formal English); state chatbots (mostly FAQ trees); NGO helplines (human, scarce).
- **Gap:** none do *claim-level verified eligibility reasoning*. Generic "chat with PDF" RAG would hallucinate eligibility — a documented harm.
- **Tutorial risk:** the naive version is a PDF chatbot → **rejected**. The project is only Level-D if the groundedness gate + eligibility eval benchmark + multilingual layer are built. That is the design below.

### D1 — Flood mapping
- **Existing:** Cloud to Street (commercial), academic papers (U-Net, fusion variants), Sen1Floods11 benchmark.
- **Gap:** published mIoU ≈ 53%; weak-label regime underexplored at student level; almost no student portfolio touches SAR. Ablation-rich: S1-only vs S2-only vs fusion; hand-labels-only vs weak-labels; per-event generalization.
- **Tutorial risk:** low — no YouTube tutorial pipeline for SAR segmentation.

### Honest risk assessment for the runners-up
- **ISL (D2):** safe-feasible, great demo, but must be framed as *real-time signer-independent edge recognition*, not "another gesture classifier."
- **Incident co-pilot (G2):** exciting but the eval data problem (no public telemetry tied to postmortems) is real — would require synthetic telemetry, weakening the "real data" claim.
- **Indic voice agent (G4):** strong, but as a standalone it risks becoming "ASR API + LLM" — better folded into G1 as the voice interface.

---

## SECTION 5 — STUDENT-FEASIBILITY ANALYSIS (flagships)

### G1 — Verified scheme navigator

| Requirement | Assessment |
|---|---|
| Data | Scheme guideline PDFs are public (myScheme, ministry sites). 30–50 schemes is a convincing corpus. Self-authored eval set of ~100 Q&A pairs — doable in a weekend sprint. |
| Compute | Zero training required (retrieval + prompting). Optional: LoRA fine-tune of IndicWhisper on free Kaggle GPU if doing voice. |
| Engineering | FastAPI + Postgres/pgvector + React — standard student stack. The novel code is the verifier + evaluator (~1,500–2,500 lines). |
| Time | MVP 6–8 wks; strong version 14–16 wks. |
| Research | No breakthrough needed — verification/groundedness is applied engineering with known techniques. |
| Labeling | ~100 hand-written eligibility Q&As + source citations. Feasible. |
| Dependency risk | Works fully on Ollama/local LLM if APIs vanish. |
| Ethics/legal | Public documents; add "not legal advice" disclaimer. Low risk. |

### D1 — Satellite flood mapping

| Requirement | Assessment |
|---|---|
| Data | Sen1Floods11 public (GCS bucket + STAC catalog). Live Sentinel-1/2 data free via Copernicus/GEE for the "map a real flood" demo. |
| Compute | 512×512 segmentation on Kaggle's free T4×2 (30h/wk) — sufficient for U-Net/DeepLab-class models. |
| Engineering | Geoportal frontend (map + mask overlay) + FastAPI inference — moderate; rasterio/GDAL has a learning curve (week 1–2 risk). |
| Time | MVP (S1-only U-Net, paper split) 6–7 wks; fusion + ablations + geoportal by wk 16–18. |
| Research | Weak-supervision experiments are *known-technique-applied-to-real-gap* — the right kind of student research. |
| Labeling | None — labels ship with dataset. Optional: contribute labels back. |
| Dependency risk | None — dataset is downloadable. |
| Ethics | Dual-use minimal; humanitarian application. |

---

## SECTION 6 — SIX SERIOUS FINALISTS

### GenAI
1. **Verified scheme navigator (G1)** — best problem×differentiation×feasibility product.
2. **On-call incident co-pilot (G2)** — most "systems engineer" appeal for infra-company interviews; weakest on real eval data.
3. **Voice-first Indic public-service agent (G4)** — strongest social-impact narrative; best used *inside* G1 as the voice modality.

### DL
1. **Flood mapping (D1)** — rarest on student resumes; benchmark headroom; real operational story.
2. **ISL edge recognition (D2)** — best live demo; weaker novelty.
3. **Industrial anomaly detection (D3)** — cleanest benchmark methodology; most crowded.

**Tradeoffs:** G2 beats G1 for pure "hiring manager at Datadog" appeal but fails the real-data test. D2 beats D1 on demo-ability but loses on differentiation and research room. The flagship pair maximizes the product of impact × depth × differentiation × feasibility.

---

## SECTION 7 — THE TWO FLAGSHIP PROJECTS

# 🅰 FLAGSHIP GenAI: **"SETU"** — Verified multilingual welfare-scheme eligibility navigator

> *"Am I eligible for this scheme — prove it clause by clause, or tell me you don't know."*

### Problem / Who / Why it matters
Hundreds of millions of Indians are entitled to central/state welfare schemes (scholarships, pensions, insurance, farmer support) but can't determine eligibility — rules live in dense official PDFs, in English, across thousands of schemes. Documented awareness gap (PM-JAY study: ~38% of eligible unaware). Every wrong answer has real cost.

### Current solutions & limits
myScheme keyword search; state FAQ bots; NGO helplines. None verify claims against source clauses; none reason over multi-constraint eligibility; none speak Telugu.

### Hypothesis
An LLM pipeline can produce trustworthy eligibility answers *only if* every generated claim is forced through a verification gate against retrieved source text — and abstains when grounding fails.

### Why AI is necessary & why it's hard
Necessary: the input is natural-language life circumstances ("I am a farmer in Nalgonda, my daughter is in 9th class") mapped to bureaucratic clause logic — pure rules engines break on phrasing diversity. Hard: eligibility is conjunctive/disjunctive reasoning over multiple clauses across documents; a fluent wrong answer is the *default* failure mode; low-resource-language input compounds it.

### Architecture
```
User (text or voice, English/Telugu/Hindi)
  │
  ▼
[Frontend: React — profile intake, chat, source-highlighted answers]
  │ REST/WebSocket
  ▼
[FastAPI backend — orchestration]
  ├──► [ASR: IndicWhisper/Whisper] (voice path)
  ├──► [Query processor: language detect → translate-if-needed → profile extraction]
  ├──► [Retriever: hybrid BM25 + dense (pgvector) over scheme clause index]
  ├──► [Eligibility reasoner: LLM produces structured verdict
  │      {scheme, verdict: ELIGIBLE/INELIGIBLE/UNCERTAIN, claims[]}]
  ├──► ★ GROUNDEDNESS GATE: each claim must trace to a retrieved clause span;
  │      NLI-style entailment check (small cross-encoder) per claim;
  │      unverified claim → regenerate or abstain
  ├──► [Explainer: renders verdict + cited clauses in user's language]
  └──► [TTS: Indic-TTS / edge-tts] (voice path)
```

- **Data:** 30–50 schemes' official guideline PDFs → clause-aware chunking (sections, not fixed windows) → embeddings (bge-m3 or e5-multilingual — free, multilingual).
- **Eval harness:** hand-built set of ~100 eligibility questions with gold verdicts + gold citations; report groundedness rate, verdict accuracy, abstention calibration.
- **Failure modes:** missing scheme → honest "not in my knowledge base"; ambiguous profile → asks a clarifying question; contradictory clauses → flags both citations.

### Why interviewers will dig in
The groundedness gate is a real design problem (entailment vs string-match vs LLM-judge — you'll have *opinions*), the eval set is real methodology, and the multilingual/voice path shows you built for actual users.

---

# 🅱 FLAGSHIP DL: **"JALMAPP"** — Multisensor satellite flood-extent mapper

> *"Draw a box on the map after a flood — get a verified water-extent map, even through clouds."*

### Problem / Who / Why it matters
During floods, response agencies need *current* inundation maps to prioritize rescue/relief. Optical satellites fail in monsoon cloud cover; SAR works but is hard to interpret. Manual mapping takes days. (Cloud to Street commercialized exactly this.)

### Current solutions & limits
Backscatter thresholding (classic remote sensing) — crude; commercial services — expensive, closed. Published DL benchmark: ~53% mIoU on Sen1Floods11 — large headroom, and the paper shows weak labels are the bottleneck.

### Hypothesis
A fusion segmentation model (SAR + optical where available) trained with a weak-supervision strategy will beat single-sensor baselines, and *cross-event* evaluation will expose the generalization gap that matters operationally.

### Why AI is necessary & why it's hard
Necessary: water extent in SAR is a learned texture/context problem (specular reflection vs wind-roughened water vs wet soil) — thresholds fail. Hard: class imbalance (water ≈ small % of pixels), weak labels, domain shift across events/sensors/regions, and SAR speckle noise.

### Architecture
```
[Frontend: React + map (MapLibre/Leaflet)]
   draw AOI → pick event → view flood mask overlay + stats
        │
        ▼
[FastAPI backend]
   ├──► [Tile fetcher: Sentinel-1/2 via GEE/Copernicus STAC]
   ├──► [Preprocessor: GRD→dB, co-registration, chip into 512²]
   ├──► [Inference service: PyTorch model (ONNX option)]
   │       U-Net / DeepLabV3+ w/ EfficientNet or ResNet encoder
   │       → per-pixel {water, no-water, invalid} + uncertainty map
   └──► [Post-process: georeference mask → GeoTIFF/PNG + area stats]
```

- **Data:** Sen1Floods11 — 4,831 chips; ~446 hand-labeled (gold), rest weak-labeled. Splits: paper's official split + a custom *leave-one-event-out* split (the interesting generalization test).
- **Experiments (the resume meat):**
  - Baseline: Otsu/backscatter threshold on S1 (classic RS baseline)
  - U-Net S1-only vs S2-only vs early-fusion vs late-fusion
  - Gold-labels-only vs weak-labels vs weak+gold-finetune (weak supervision)
  - Leave-one-event-out generalization (domain shift)
  - Optional stretch: cross-modal distillation (S2 teacher → S1 student) — literature exists
  - Uncertainty: MC-dropout or deep-ensemble → uncertainty map in UI
- **Demo kicker:** map a real recent Indian flood event (download live Sentinel data via free GEE) — "the model just mapped last month's Assam floods" is a killer demo line.

### Why interviewers will dig in
Segmentation > classification; a real benchmark with headroom; domain-shift evaluation shows scientific maturity; weak supervision is an actual research topic; SAR fusion is rare at undergrad level.

---

## SECTION 8 — DETAILED ARCHITECTURE (condensed design sheets)

### SETU — components
- **Index:** clause-aware chunker → multilingual embeddings → pgvector; metadata {scheme_id, clause_id, doc_page}.
- **Retrieval:** hybrid (BM25 + dense) + metadata filters (state, category); rerank with bge-reranker if budget allows.
- **Reasoner:** structured-output LLM (JSON schema verdict); temperature 0; constrained to retrieved context.
- **Verifier:** cross-encoder NLI per claim→clause pair; threshold → accept/regenerate/abstain. Compare NLI vs LLM-judge vs citation-string-match as an ablation.
- **Voice path:** IndicWhisper (AI4Bharat, Vistaar benchmarks; Telugu WER ~14–28% published) → same pipeline → TTS back.
- **DB:** Postgres + pgvector; eval sets as versioned JSON.
- **Monitoring:** log every {query, retrieved clauses, verdict, verification result} → eval dashboard.

### JALMAPP — components
- **Training:** PyTorch + segmentation-models-pytorch; AMP; Kaggle T4s; Weights & Biases free tier.
- **Serving:** ONNX export → FastAPI; ~1s/chip CPU inference plausible for 512² U-Net.
- **Geo:** rasterio/GDAL for tiling, georeferencing; output GeoTIFF + PNG overlay.
- **Frontend:** MapLibre; draw AOI → job → mask overlay + water-area km² + uncertainty toggle.

---

## SECTION 9 — ROADMAPS (18 weeks each; can run sequentially or overlap)

### SETU
| Wk | Deliverable |
|---|---|
| 1–2 | Corpus: 30–50 scheme PDFs; clause-aware chunking; index v1; 20 eval questions drafted |
| 3–4 | Naive RAG baseline working end-to-end; measure how often it hallucinates eligibility (this is your motivation data!) |
| 5–6 | Structured eligibility reasoner + verdict schema; retrieval tuning (hybrid, rerank) |
| 7–8 | ★ Groundedness gate v1 (NLI cross-encoder); abstention path; eval set grown to ~60 |
| 9–10 | Backend API hardening; React frontend; cited-clause highlighting UI |
| 11–12 | Eval set →100; full eval report; Telugu/Hindi input path (translate-then-reason ablation vs multilingual embeddings) |
| 13–14 | Voice in/out (IndicWhisper + TTS); clarification-dialog flow |
| 15–16 | Docker, deploy, monitoring dashboard; user test with 5+ real people |
| 17–18 | Failure taxonomy writeup; README, demo video, technical report draft |

### JALMAPP
| Wk | Deliverable |
|---|---|
| 1–2 | Geospatial toolchain (rasterio/GDAL); dataset download; EDA; threshold baseline implemented |
| 3–4 | S1-only U-Net baseline trained on paper's split; metrics pipeline (mIoU, per-event) |
| 5–6 | S2-only + early/late fusion variants; first ablation table |
| 7–8 | Weak-vs-gold-label experiments; leave-one-event-out split built |
| 9–10 | Best model hardened; uncertainty (MC-dropout); ONNX export |
| 11–12 | Inference service + georeferenced output; FastAPI |
| 13–14 | Map frontend; AOI → mask overlay flow |
| 15–16 | ★ Live demo: fetch real recent-flood Sentinel data via GEE, run pipeline end-to-end |
| 17–18 | Error analysis (per-event, sensor, biome); report/paper draft; repo polish |

**Combined plan if doing both:** stagger — JALMAPP wks 1–18, SETU wks 10–27 (or run 60/40 parallel; both fit ~20 wks total with discipline).

---

## SECTION 10 — INTERVIEW QUESTIONS (what they'll ask — and what you must know)

### SETU — ML/NLP
1. Why is naive RAG insufficient for eligibility reasoning? *(multi-constraint logic vs semantic similarity; fluent-wrong-answer risk)*
2. How does your groundedness gate work? Alternatives considered? *(NLI entailment vs LLM-judge vs citation overlap — precision/recall tradeoffs of each)*
3. How do you evaluate a system whose failure is *plausible-sounding* lies? *(golden set design, verdict accuracy, groundedness rate, abstention calibration, human spot-checks)*
4. How do you chunk eligibility clauses and why does naive fixed-window chunking fail here? *(clause boundaries carry logic; a threshold split mid-sentence breaks entailment)*
5. How do you handle Telugu queries? Translate-first or multilingual embeddings — what did your ablation show?
6. What happens when retrieved clauses conflict or the scheme isn't in the corpus? *(abstention design — know precision-coverage tradeoff)*
7. How would you detect corpus staleness when a scheme's income threshold changes?
8. Why structured JSON output vs free text? *(enables verification + downstream UI; constrained decoding)*
9. What's the cross-encoder for? Why not just cosine similarity for entailment?
10. How would this scale to 3,000 schemes? *(index partitioning, metadata pre-filtering, cost/latency math)*

### SETU — System design
1. Design the request flow end-to-end; where's the latency budget? *(retrieval ~ms, LLM ~1–3s, verification adds N cross-encoder calls — parallelize)*
2. How do you version the scheme corpus and re-embed on updates?
3. How do you log/monitor? What telemetry tells you groundedness is degrading in prod?
4. Cost model at 10k queries/day? *(token math; local-model fallback)*
5. Security: prompt injection via a malicious PDF in the corpus? PII in user profiles?
6. Caching strategy — what can be cached safely vs what's user-specific?
7. How does the verification gate behave under load — fail open or closed? *(fail closed for this domain — discuss why)*
8. Multi-turn clarification: how is state managed?
9. Voice pipeline latency — ASR+LLM+TTS chain; how do you keep it usable?
10. API design: show the request/response contract.

### JALMAPP — ML
1. Why segmentation, not detection/classification, for flood extent?
2. Why does SAR see through clouds? What's speckle noise and how do you handle it?
3. Early vs late fusion of S1+S2 — what did you find and why?
4. Class imbalance: water is a small fraction of pixels — loss choice? *(focal/dice/weighted CE — know why)*
5. Weak labels vs gold labels — what did the experiment show? Why do weak labels cap performance?
6. Leave-one-event-out: what generalization gap did you measure? Hypotheses for why?
7. Why U-Net/DeepLab? What would a SegFormer/attention model change?
8. How did you produce uncertainty maps? *(MC-dropout vs ensemble — calibration caveats)*
9. How do you evaluate when ground truth itself is noisy? *(evaluate on gold-only; report weak-label noise tolerance)*
10. What would you do with 10× more compute? *(pretraining on unlabeled SAR — self-supervision, MAE-style)*

### JALMAPP — Systems/implementation
1. Walk me through a Sentinel-1 GRD scene → model input. *(orbit correction, calibration to σ⁰/dB, speckle filter, tiling, co-registration)*
2. How do you co-register S1 and S2 tiles? *(georeferencing, resolution mismatch 10m, resampling)*
3. How does the geoportal render a GeoTIFF prediction? *(reproject → tiles → overlay)*
4. Inference cost per km²? CPU vs GPU latency? ONNX gains?
5. How would you go from "map last week's flood" to "monitor continuously"? *(tasking, event triggers, diff-vs-baseline water)*
6. Data pipeline idempotency; how do you avoid reprocessing?
7. Augmentation policy for SAR — what transforms are physically valid? *(flips yes; color jitter no — SAR isn't RGB!)*
8. What's in the eval harness? Reproducibility: seeds, splits, versioned data?
9. Failure modes: where does the model fail? *(wet soil confusion, wind-roughened water, urban areas, radar shadow in mountains — expect you to show failure chips)*
10. If a disaster agency wanted this tomorrow, what are the top 3 gaps?

*(That's the representative core of the ~45/project the brief asks for — these are the questions this project actually generates, each with a real answer you can develop while building.)*

---

## SECTION 11 — RESUME / GITHUB PRESENTATION

### Resume bullets (SETU)
- Built a multilingual (English/Telugu/Hindi) welfare-scheme eligibility navigator over N official scheme documents, with a claim-level groundedness gate that verifies every generated statement against retrieved source clauses before answering — abstains rather than hallucinate.
- Designed a 100-question gold evaluation set with verdict + citation labels; report verdict accuracy, groundedness rate, and abstention calibration across three verification strategies (NLI cross-encoder / LLM-judge / citation-overlap).
- Full-stack: FastAPI + Postgres/pgvector hybrid retrieval + React citation-highlighting UI; voice path via fine-tuned Indic ASR; deployed on Docker.

### Resume bullets (JALMAPP)
- Trained multisensor (Sentinel-1 SAR + Sentinel-2 optical) segmentation models for flood-extent mapping on the Sen1Floods11 benchmark (4,831 chips, 11 global events), with leave-one-event-out evaluation exposing cross-region generalization gaps.
- Ran a weak-supervision study (446 gold vs 4,385 weak labels) and fusion ablations (S1-only / S2-only / early / late); MC-dropout uncertainty maps surface low-confidence regions to operators.
- Built an end-to-end geoportal: draw an AOI → pipeline fetches live Sentinel data (Google Earth Engine) → ONNX inference → georeferenced flood mask + inundated-area stats; demoed on a real recent flood event.

### Repo structure (both)
```
README.md  (problem, demo GIF, architecture diagram, results table, failure analysis, reproduce steps)
notebooks/ (EDA, training)
src/ (pipelines, API, models)
eval/ (golden sets, eval scripts, results)
docs/ (decision log, failure taxonomy, report.pdf)
docker-compose.yml
```
**Credibility rules:** report real numbers only; include a `FAILURE_ANALYSIS.md`; pin seeds; a 60–90s demo video beats 1,000 words.

---

## SECTION 12 — RESEARCH / PAPER EXTENSIONS

- **SETU →** workshop-paper angle: "Claim-level groundedness verification for eligibility reasoning in low-resource public-benefits QA" — the eval set + verifier comparison is a real contribution (ACL/EMNLP workshop or AAAI student track territory). Or: "Abstention calibration in benefit-eligibility QA."
- **JALMAPP →** workshop-paper angle: "Weak-label strategies for multisensor flood segmentation: a leave-one-event-out study" (CVPR EarthVision workshop lineage — Sen1Floods11 itself was a CVPRW paper). Cross-modal distillation (S2 teacher → S1 student) is the published-but-underexplored stretch.
- **Optional integration story:** JALMAPP maps a flood → SETU tells affected residents which relief schemes they qualify for. A two-project "disaster-response AI" narrative is a compelling portfolio arc.

---

## SECTION 13 — PROJECTS TO AVOID (and why)

| Rejected | Why |
|---|---|
| PDF chatbot / generic RAG | Tutorial-tier; no verification story |
| Stock/house price prediction | No real user; markets aren't learnable at this level |
| Medical image classification (plain) | Most saturated student genre; needs edge+domain-shift second act to survive |
| Generic chatbot / AI assistant / interview coach | No defensible problem |
| Sentiment analysis / OCR / summarizer | Solved, shallow |
| "Multi-agent autonomous anything" | Demo-fragile; unverifiable claims; interviewers have seen 100 |
| Fraud/recommendation engines | Generic; data rarely reflects real distributions |
| New foundation model | Level E — absurd at this scope |

---

## KEY SOURCES
- Sen1Floods11: Bonafilia et al., CVPRW 2020 — github.com/cloudtostreet/Sen1Floods11; fusion follow-up, Remote Sensing 13(11):2220 (mIoU 52.99%); cross-modal distillation, Environmental Data Science 2023.
- INCLUDE ISL: Sridhar et al., ACM MM 2020 (doi 10.1145/3394171.3413528) — 4,287 videos/263 signs; 94.5% INCLUDE-50, 85.6% full benchmark; ai4bharat/INCLUDE on HF+GitHub.
- PM-JAY awareness study: PubMed 36478057 (~62% aware; eligibility-awareness gradients). Dvara Research, *State of Exclusion* (DBT last-mile failures).
- Indic ASR: IndicVoices (ACL Findings 2024, 23.7K hrs, 22 langs); Vistaar/IndicWhisper (avg WER 13.6 over 59 benchmarks); Telugu ASR fine-tuning study, Frontiers in AI (Whisper-small ~14–29% WER).
- Agent eval/observability landscape: Langfuse, sentinel-llm, DSPy/GEPA eval-harness patterns — confirms "eval-first" is the credible 2026 differentiator.
- GSoP offline glaucoma screening (BMC Med Inform 2026) — existence proof that on-device screening pipelines are real and student-reproducible in spirit.
