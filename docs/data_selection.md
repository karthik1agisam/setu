# Scheme selection — Phase 1 decision record

## Rubric (from master plan, Part 8)

Each candidate scored 0–2 on:
(a) official doc exists on a government domain; (b) machine-readable text
layer; (c) explicit eligibility rules; (d) rule structure differs from other
selected schemes; (e) document carries a date/version.

## Selected corpus

| Scheme | Docs | Rubric | Notes |
|---|---|---|---|
| **PM-KISAN** | Operational Guidelines (rev. 21.06.2019) + FAQ | 10/10 | Exclusion-list structure; real revision history (2 ha cap removed June 2019) enables outdated-policy benchmark traps |
| **PM-JAY** | Beneficiary Empowerment Guidebook (2019) | 9/10 | SECC-2011 deprivation/occupation *category membership* — different structure from threshold rules. pmjay.gov.in bot-blocks scripts → retrieved identical file via archive.org |
| **PMAY-G** | Framework for Implementation, 2022 | 9/10 | SECC + 13-point exclusion list + Gram-Sabha verification step — adds a process-layer condition. 161 pages, rich clause structure. pmayg.nic.in unreachable → archive.org mirror |
| **PMS-SC** | Post-Matric Scholarship guidelines (2020-21→2025-26) | 9/10 | Numeric thresholds (income ≤ ₹2.5 lakh, category, matric pass) — clean multi-condition reasoning. Canonical ministry PDF is *scanned* (0-char text layer); identical text retrieved from Gujarat SJE dept. mirror (official .gov.in) |

## Rejected / deferred candidates

| Candidate | Reason |
|---|---|
| Telangana Aasara / Rythu Bharosa | No single clean official guideline PDF located quickly; deferred — can be added later via the same ingestion pipeline |
| PMVVY | Scheme window status unclear; PM-SYM similar |
| NSAP (IGNOAPS) | Fine backup if any selected scheme proves unworkable during benchmark authoring |

## Provenance

All downloads recorded in `data/raw/manifest.csv` — canonical `source_url`,
actual `retrieval_url` (mirror/archive used honestly recorded), doc version,
retrieval date, SHA-256, page count, extractable char count.

## Known data issues found in Phase 1

- `pmjay.gov.in` and `pmayg.nic.in` reject scripted access → archive.org
  `id_` raw-content endpoints used; same files, provenance recorded.
- The canonical PMS-SC PDF on `socialjustice.gov.in` and
  `scholarships.gov.in` is a **scanned image** (0 extractable chars).
  The Gujarat SJE republished copy has a full text layer and identical
  content.
