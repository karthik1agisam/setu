"""Unit tests for ingestion stages on synthetic fixtures."""

from ai.ingestion.clean import clean_pages, dehyphenate, find_boilerplate
from ai.ingestion.extract import PageText
from ai.ingestion.structure import parse_blocks


def _pages(n: int, header: str = "PM-KISAN OPERATIONAL GUIDELINES") -> list[PageText]:
    pages = []
    for i in range(1, n + 1):
        lines = [header, f"content line {i}", f"unique content {i}", str(i)]
        pages.append(PageText(page=i, lines=lines))
    return pages


def test_find_boilerplate_catches_repeated_header():
    pages = _pages(10)
    boiler = find_boilerplate(pages)
    assert "PM-KISAN OPERATIONAL GUIDELINES" in boiler
    assert "unique content 3" not in boiler


def test_clean_removes_header_and_page_number():
    pages = clean_pages(_pages(10))
    for p in pages:
        assert "PM-KISAN OPERATIONAL GUIDELINES" not in p.lines
        assert all(not ln.isdigit() for ln in p.lines)


def test_dehyphenate():
    assert dehyphenate("culti-\nvable land") == "cultivable land"


def test_structure_numbered_hierarchy():
    pages = [
        PageText(
            page=1,
            lines=[
                "3. Definition of farmer's family",
                "A landholder farmer's family is defined as...",
                "4 Exclusions",
                "4.1 The following categories shall not be eligible:",
                "(a) All Institutional Land holders; and",
                "(b) Farmer families in which one or more members belong to:",
                "(i) Former and present holders of constitutional posts",
                "(ii) Former and present Ministers",
            ],
        )
    ]
    blocks = parse_blocks(pages)
    paths = [b.section_path for b in blocks]
    assert "3" in paths
    assert "4 > 4.1" in paths
    assert "4 > 4.1 > (a)" in paths
    assert "4 > 4.1 > (b) > (i)" in paths
    assert "4 > 4.1 > (b) > (ii)" in paths


def test_structure_standalone_number_line():
    """PDFs often render the clause number alone on a line (real corpus bug:
    PMS-SC section 5 was silently absorbed into section 4)."""
    pages = [
        PageText(
            page=1,
            lines=[
                "4. Eligibility of Institutions",
                "4.1 Institutions must be recognized.",
                "5.",
                "Conditions of Eligibility of Students",
                "5.1",
                "The scholarships are open to Nationals of India only.",
            ],
        )
    ]
    blocks = parse_blocks(pages)
    paths = [b.section_path for b in blocks]
    texts = " ".join(b.text for b in blocks)
    assert "4" in paths
    assert "5" in paths
    assert "5 > 5.1" in paths
    assert "Nationals of India" in texts


def test_structure_preamble_when_no_numbering():
    pages = [PageText(page=1, lines=["Just some intro text.", "More text."])]
    blocks = parse_blocks(pages)
    assert len(blocks) == 1
    assert blocks[0].section_path == ""
    assert "intro text" in blocks[0].text
