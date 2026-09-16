#!/usr/bin/env python3
"""Download official scheme documents and record provenance.

For each document: download (with mirror/archive fallback) → validate it is a
machine-readable PDF → record sha256/pages/dates in data/raw/manifest.csv.

Idempotent: existing files are re-validated, not re-downloaded.
Re-run safe. Exit code 0 only if every document is present AND
machine-readable.

Usage:  uv run python scripts/download_docs.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
MANIFEST = RAW_DIR / "manifest.csv"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
TIMEOUT = 90
MIN_EXTRACTABLE_CHARS = 500  # a real text layer yields far more than this


@dataclass(frozen=True)
class Doc:
    scheme: str
    filename: str
    source_url: str  # canonical official URL (recorded as provenance)
    retrieval_urls: tuple[str, ...]  # mirrors/archives tried in order
    doc_version: str  # document date/version as printed on the document


DOCS: list[Doc] = [
    Doc(
        scheme="pmkisan",
        filename="operational_guidelines_2019.pdf",
        source_url="https://www.pmkisan.gov.in/Documents/Revised%20Operational%20Guidelines%20-%20PM-Kisan%20Scheme.pdf",
        retrieval_urls=(
            "https://www.pmkisan.gov.in/Documents/Revised%20Operational%20Guidelines%20-%20PM-Kisan%20Scheme.pdf",
        ),
        doc_version="Revised operational guidelines, as on 21.06.2019",
    ),
    Doc(
        scheme="pmkisan",
        filename="faq.pdf",
        source_url="https://pmkisan.gov.in/Documents/FAQPMKISAN.pdf",
        retrieval_urls=("https://pmkisan.gov.in/Documents/FAQPMKISAN.pdf",),
        doc_version="FAQ (undated, official portal)",
    ),
    Doc(
        scheme="pmjay",
        filename="beneficiary_empowerment_guidebook.pdf",
        source_url="https://www.pmjay.gov.in/sites/default/files/2019-03/Beneficiary%20Empowerment%20Guidebook.pdf",
        retrieval_urls=(
            "https://www.pmjay.gov.in/sites/default/files/2019-03/Beneficiary%20Empowerment%20Guidebook.pdf",
            # archive.org raw-content endpoint (id_) — pmjay.gov.in bot-blocks scripts
            "https://web.archive.org/web/2020id_/https://www.pmjay.gov.in/sites/default/files/2019-03/Beneficiary%20Empowerment%20Guidebook.pdf",
        ),
        doc_version="Beneficiary Empowerment Guidebook (2019)",
    ),
    Doc(
        scheme="pmayg",
        filename="framework_for_implementation_2022.pdf",
        source_url="https://pmayg.nic.in/netiay/writereaddata/Circulars/Framework%20For%20Implementation%20(FFI)%20of%20PMAY-G%20(2022).pdf",
        retrieval_urls=(
            "https://pmayg.nic.in/netiay/writereaddata/Circulars/Framework%20For%20Implementation%20(FFI)%20of%20PMAY-G%20(2022).pdf",
            # archive.org raw-content endpoint (id_) — pmayg.nic.in unreachable from scripts
            "https://web.archive.org/web/2023id_/https://pmayg.nic.in/netiay/writereaddata/Circulars/Framework%20For%20Implementation%20(FFI)%20of%20PMAY-G%20(2022).pdf",
        ),
        doc_version="Framework for Implementation, 2022 revision",
    ),
    Doc(
        scheme="pmssc",
        filename="scheme_guidelines.pdf",
        source_url="https://socialjustice.gov.in/writereaddata/UploadFile/PMS_for_SCs_Scheme_Guidelines.pdf",
        retrieval_urls=(
            # canonical ministry PDF is a scanned image (0-char text layer);
            # identical guidelines republished by Gujarat SJE dept are machine-readable
            "https://sje.gujarat.gov.in/dscw/assets/downloads/gr6_dscw_sje_02092022.pdf",
            "https://socialjustice.gov.in/writereaddata/UploadFile/PMS_for_SCs_Scheme_Guidelines.pdf",
        ),
        doc_version="PMS-SC scheme guidelines, effective 2020-21 to 2025-26",
    ),
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
    except Exception as e:  # noqa: BLE001 — report and try next mirror
        print(f"    fetch failed: {e}")
        return False
    if len(data) < 10_000 or not data.startswith(b"%PDF"):
        print(f"    rejected: {len(data)} bytes, not a PDF")
        return False
    dest.write_bytes(data)
    return True


def validate_pdf(path: Path) -> tuple[int, int]:
    """Return (pages, extractable_chars). Raises on unparseable PDF."""
    doc = pymupdf.open(path)
    chars = sum(len(p.get_text()) for p in doc)
    pages = doc.page_count
    doc.close()
    return pages, chars


def main() -> int:
    rows: list[dict] = []
    failures = 0

    for d in DOCS:
        dest = RAW_DIR / d.scheme / d.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{d.scheme}] {d.filename}")

        retrieved_via = "existing"
        if not dest.exists():
            ok = False
            for url in d.retrieval_urls:
                print(f"    GET {url[:95]}")
                if fetch(url, dest):
                    retrieved_via = url
                    ok = True
                    break
            if not ok:
                print("    FAILED — all sources exhausted")
                failures += 1
                continue
        else:
            print("    exists — validating")

        try:
            pages, chars = validate_pdf(dest)
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED — unparseable PDF: {e}")
            failures += 1
            dest.unlink(missing_ok=True)
            continue

        digest = sha256_of(dest)
        machine_readable = chars >= MIN_EXTRACTABLE_CHARS
        status = "ok" if machine_readable else "SCANNED — no text layer"
        print(f"    {pages} pages, {chars} chars extractable, sha256 {digest[:12]}… [{status}]")

        rows.append(
            {
                "scheme": d.scheme,
                "filename": d.filename,
                "source_url": d.source_url,
                "retrieval_url": retrieved_via,
                "doc_version": d.doc_version,
                "retrieval_date": date.today().isoformat(),
                "sha256": digest,
                "pages": pages,
                "extractable_chars": chars,
                "machine_readable": machine_readable,
            }
        )
        if not machine_readable:
            failures += 1

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["machine_readable"])
    print(f"\n{ok}/{len(DOCS)} documents machine-readable; manifest → {MANIFEST}")
    if failures:
        print(f"{failures} document(s) FAILED or unreadable — see above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
