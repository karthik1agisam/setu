"""Unit tests for block→chunk sizing logic."""

from ai.ingestion.chunk import chunk_blocks

META = {
    "scheme": "t",
    "doc": "d.pdf",
    "doc_version": "v1",
    "source_url": "https://x.gov.in/d.pdf",
    "section_path": "4 > 4.1",
    "page_start": 3,
    "page_end": 3,
}


def _blk(text: str, path: str = "4 > 4.1", page: int = 3) -> dict:
    return {
        **META,
        "section_path": path,
        "page_start": page,
        "page_end": page,
        "text": text,
    }


def test_short_siblings_merge():
    blocks = [_blk(f"Clause item {i} is short.") for i in range(30)]
    chunks = chunk_blocks(blocks, "t:doc")
    # 30 × ~6-word clauses should merge into few chunks
    assert 1 <= len(chunks) <= 3
    assert all(c.n_words >= 30 for c in chunks[:-1])


def test_long_clause_splits_at_sentences():
    sent = "The applicant must satisfy this particular eligibility condition. "
    blocks = [_blk(sent * 120)]  # ~1200 words, must split
    chunks = chunk_blocks(blocks, "t:doc")
    assert len(chunks) >= 3
    assert all(c.text.rstrip().endswith(".") for c in chunks)  # never mid-sentence
    assert all(c.section_paths == ["4 > 4.1"] for c in chunks)


def test_provenance_preserved_on_split_and_merge():
    blocks = [
        _blk("Short one.", path="4 > 4.1 > (a)", page=5),
        _blk("Short two.", path="4 > 4.1 > (b)", page=6),
        _blk("word " * 400, path="5", page=7),
    ]
    chunks = chunk_blocks(blocks, "t:doc")
    merged = chunks[0]
    assert merged.page_start == 5 and merged.page_end == 6
    assert merged.source_url == META["source_url"]
    big = chunks[-1]
    assert all(c.doc_version == "v1" for c in chunks)
    assert big.n_words > 0


def test_single_chunk_when_medium():
    blocks = [_blk("word " * 200)]
    chunks = chunk_blocks(blocks, "t:doc")
    assert len(chunks) == 1
