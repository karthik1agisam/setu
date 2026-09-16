"""Structure parsing: flat cleaned lines → clause blocks with section paths.

Handles the numbering conventions used by Indian government guidelines:
  4.        Section headings
  4.1       Sub-sections
  (a) (b)   Lettered items
  (i) (ii)  Roman items
  i) ii)    Bare-roman items (common inside lettered items)

Each block = one clause/sub-clause with its hierarchical section_path,
e.g. "4 > 4.1 > (b) > (iii)". Lines before any detected structure become a
"preamble" block. Non-structural lines attach to the current clause.

Ambiguity note: "(i)", "(v)", "(x)" are both valid letters and valid roman
numerals. Resolution rule — a single ambiguous char counts as a letter only
when it is the expected successor in the current letter sequence
(e.g. "(i)" after "(h)"); otherwise it is roman. Multi-char "(iv)", "(iii)"
and chars {c,d,l,m} are treated as roman/letter respectively (documents do
not have 50–500-item roman lists, and multi-char romans are never letters).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai.ingestion.extract import PageText

NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(?=\S)")
DATE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}\b")  # 01.12.2018 is a date, not a section
LETTER_RE = re.compile(r"^\(([a-z])\)\s+(?=\S)")
ROMAN_RE = re.compile(r"^\(([ivx]{2,})\)\s+(?=\S)", re.IGNORECASE)  # multi-char only
BARE_ROMAN_RE = re.compile(r"^([ivx]{1,4})[.)]\s+(?=\S)", re.IGNORECASE)
AMBIGUOUS = {"i", "v", "x"}  # chars that are both letters and roman numerals


STANDALONE_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?$")  # "5." or "5.1" alone on a line


@dataclass
class Block:
    page: int  # page where the block starts
    page_end: int  # page where the block ends
    section_path: str  # "4 > 4.1 > (b) > (iii)"; "" = preamble
    text: str


def _number_depth(num: str) -> int:
    return num.count(".") + 1


def _merge_standalone_numbers(lines: list[str]) -> list[str]:
    """Join a line that is only a clause number ("5.", "5.1") with the next line.

    Government PDFs often render the clause number on its own line — without
    this, the number attaches to the previous clause and its content is
    silently absorbed into the wrong block.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        if STANDALONE_NUM_RE.match(lines[i]) and i + 1 < len(lines):
            out.append(f"{lines[i]} {lines[i + 1]}")
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return out


def parse_blocks(pages: list[PageText]) -> list[Block]:
    blocks: list[Block] = []
    path: list[str] = []  # one label per depth level
    buf: list[str] = []
    buf_page = 1
    buf_end = 1
    have_buf = False
    # expected next letter per depth, e.g. after "(b)" at depth d → {d: "c"}
    expected_letter: dict[int, str] = {}

    def flush() -> None:
        nonlocal have_buf, buf
        if have_buf:
            # collapse empty gap components — a clause like "6.2.1" may arrive
            # without separately-detected "6"/"6.2" heading lines; the depth is
            # already encoded in the label itself
            sp = " > ".join(c for c in path if c)
            blocks.append(
                Block(page=buf_page, page_end=buf_end, section_path=sp, text=" ".join(buf))
            )
            buf = []
            have_buf = False

    def push(level: int, label: str) -> None:
        del path[level:]
        while len(path) < level:
            path.append("")
        path.append(label)
        # deeper levels reset
        for k in list(expected_letter):
            if k > level:
                del expected_letter[k]

    for p in pages:
        for ln in _merge_standalone_numbers(p.lines):
            m_num = NUM_RE.match(ln)
            if m_num and DATE_RE.match(ln):
                m_num = None  # date literal, not a clause number
            m_letter = LETTER_RE.match(ln)
            m_roman = ROMAN_RE.match(ln)
            m_bare = BARE_ROMAN_RE.match(ln)

            # resolve letter-vs-roman ambiguity via sequence expectation
            is_letter = False
            is_roman = False
            if m_letter:
                ch = m_letter.group(1).lower()
                if ch not in AMBIGUOUS:
                    is_letter = True
                else:
                    lvl = _deepest_num_level(path) + 1
                    is_letter = expected_letter.get(lvl) == ch
                    is_roman = not is_letter
            elif m_roman or (m_bare and len(path) >= 2):
                is_roman = True

            if m_num:
                flush()
                buf_page = p.page
                depth = _number_depth(m_num.group(1))
                push(depth - 1, m_num.group(1))
                del path[depth:]  # new numbered clause clears letter/roman levels
                buf.append(ln)
                have_buf = True
            elif is_letter and m_letter:
                flush()
                buf_page = p.page
                depth = _deepest_num_level(path) + 1
                ch = m_letter.group(1).lower()
                push(depth, f"({ch})")
                expected_letter[depth] = chr(ord(ch) + 1)
                buf.append(ln)
                have_buf = True
            elif is_roman:
                g = (m_roman or m_bare or m_letter)
                label = f"({g.group(1).lower()})"  # type: ignore[union-attr]
                flush()
                buf_page = p.page
                depth = _deepest_num_level(path) + 2
                push(depth, label)
                buf.append(ln)
                have_buf = True
            else:
                if not have_buf:
                    buf_page = p.page
                buf.append(ln)
                have_buf = True
            buf_end = p.page
    flush()
    return blocks


def _deepest_num_level(path: list[str]) -> int:
    """Index of the deepest numeric component, or -1."""
    for i in range(len(path) - 1, -1, -1):
        if path[i] and path[i][0].isdigit():
            return i
    return -1
