"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================


from collections import defaultdict
from dataclasses import dataclass

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# A compact, case-insensitive record of company names seen in document headings.
_company_names: dict[str, str] = {}
_YEAR = re.compile(r"(?<![A-Za-z0-9])(?:FY\s*)?((?:19|20)\d{2})(?:[AE])?\b", re.I)
_METADATA_TITLES = (
    ("annual_report", re.compile(r"\bannual report\s+((?:19|20)\d{2})\b", re.I)),
    ("earnings_call", re.compile(r"\b((?:19|20)\d{2})\s+earnings\s+(?:conference\s+)?call\b", re.I)),
    ("research_report", re.compile(r"\bFY\s*((?:19|20)\d{2})\s+results review\b", re.I)),
)
_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z\d.'’&-]*(?:\s+(?:(?:and|of|the|&)\s+)?[A-Z][a-zA-Z\d.'’&-]*)*")
_QUESTION_START = re.compile(r"^(?:(?:What|Which|Who|How|Where|In|On|At|As|For|Of|Compare|Between|Did|Does|The)(?:\s+|$))+")
_STOP = frozenset("""
a an and are as at be been between by can compared compare could date december
did do does dollars end ended fiscal for from generate generated give gives had
has have held higher hold how in is it its last lower many million millions more
much of on or per percent percentage report reported s set than that the their
to usd value was were what when where which who with year vs versus
financial statement statements company companies segment firm
""".split())

# Each rule links common analyst phrasing to a source phrase or a table row.
_FIELDS = (
    ("operating_cash", r"net cash (?:from|provided by) operating activit\w*|operating cash flow",
     r"net cash (?:from|provided by) operating activit\w*|operating cash flow"),
    ("cash", r"cash and cash equivalents|\bcash\b", r"cash and cash equivalents"),
    ("receivables", r"(?:net )?accounts receivables?(?: net)?", r"accounts receivables? net|net accounts receivables?"),
    ("borrowings", r"long term (?:borrowing\w*|debt)", r"long term (?:borrowing\w*|debt)"),
    ("buyback", r"share repurchase(?: programme| program)?|buyback", r"share repurchase|buyback"),
    ("net_income", r"net (?:income|profit)", r"net income|net profit"),
    ("operating_income", r"operating (?:income|profit)", r"operating income|operating profit"),
    ("gross_profit", r"gross profit", r"gross profit"),
    ("assets", r"total assets", r"total assets"),
    ("equity", r"total (?:shareholders? )?equity", r"total (?:shareholders? )?equity"),
    ("inventories", r"\binventor(?:y|ies)\b", r"inventor(?:y|ies)"),
    ("employees", r"\bemployees?\b|headcount|workforce", r"total employees|employ(?:ed|s)|headcount|workforce"),
    ("cfo", r"chief financial officer|\bcfo\b", r"chief financial officer|\bcfo\b"),
    ("ceo", r"chief executive officer|\bceo\b", r"chief executive officer|\bceo\b"),
    ("auditor", r"audit firm|auditor|audited", r"independent auditor|audited|audit firm|statutory auditor"),
    ("hq", r"headquarter\w*|head office", r"headquarter\w*|head office"),
    ("price", r"target price|price target", r"target price|price target"),
    ("rating", r"\brating\b|recommendation", r"\brating\b|recommendation"),
    ("growth", r"revenue growth|sales growth", r"revenue growth|sales growth"),
    ("revenue", r"\b(?:revenue|sales)\b", r"(?:total )?revenue|sales"),
)
_PERSON_OR_OFFICE = {"cfo", "ceo", "auditor", "hq"}


def _text(tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


def _metadata(source, soup) -> dict:
    # Use the cover title and subtitle, so the report's broker is not its company.
    title = soup.find("h1")
    company = re.sub(r"\s*\([^()]*\)\s*$", "", _text(title)) if title else ""
    intro = " ".join(_text(tag) for tag in soup.find_all(["title", "h2", "p"], limit=7))
    for doc_type, pattern in _METADATA_TITLES:
        match = pattern.search(intro)
        if match:
            year = int(match.group(1))
            break
    else:
        doc_type = "annual_report"
        match = _YEAR.search(intro)
        year = int(match.group(1)) if match else None
    if company:
        _company_names[company_key(company)] = company
    return {"source": source, "company": company, "fiscal_year": year, "doc_type": doc_type}


def _table_rows(table):
    """Expand cells to their grid positions so colspan headers still line up."""
    carry = {}
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        row = {col: cell for col, (cell, _) in carry.items()}
        carry = {col: (cell, n - 1) for col, (cell, n) in carry.items() if n > 1}
        col = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while col in row:
                col += 1
            for i in range(col, col + int(cell.get("colspan", 1))):
                row[i] = cell
                span = int(cell.get("rowspan", 1))
                if span > 1:
                    carry[i] = (cell, span - 1)
            col += int(cell.get("colspan", 1))
        yield tr, row


def _split_plain(file_name, html, render_tables=False):
    """Use the same recursive splitter for the naive and serialized baselines."""
    soup = BeautifulSoup(html, "html.parser")
    meta = _metadata(file_name, soup)
    if render_tables:
        for table in soup.find_all("table"):
            table.replace_with("\n" + table_to_text(str(table)) + "\n")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1250, chunk_overlap=160)
    return splitter.create_documents([soup.get_text(" ", strip=True)], metadatas=[meta])


def _sized_blocks(parts, heading=""):
    """Pack adjacent source lines; repeat the table caption when splitting a table."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1250, chunk_overlap=160)
    prefix = (heading + "\n") if heading and len(heading) <= 240 else ""
    block = prefix
    for part in parts:
        if len(prefix) + len(part) > MAX_CHUNK_CHARS:
            pieces = splitter.split_text(part)
        else:
            pieces = [part]
        for piece in pieces:
            if len(block) + len(piece) + 1 > MAX_CHUNK_CHARS:
                if block != prefix:
                    yield block.rstrip()
                block = prefix
            block += piece + "\n"
    if block != prefix:
        yield block.rstrip()


def _query_type(question: str) -> str:
    if re.search(r"\b(?:earnings\s+(?:conference\s+)?call|conference call|(?:q[1-4]|quarter) call)\b", question, re.I):
        return "earnings_call"
    if re.search(r"\b(?:rating|target price|price target|forecast\w*|estimate\w*|results review|broker\w*|research report|recommendation)\b", question, re.I):
        return "research_report"
    return "annual_report"


def _clean_entity(value):
    value = _QUESTION_START.sub("", value.strip())
    value = re.sub(r"['’]s?$", "", value)
    return value.strip(" ,:;?.") or ""


def _entity_candidates(question):
    """Read complete name phrases, including legal suffixes and unknown companies."""
    names = []
    for m in _CAPITALIZED.finditer(question):
        raw = m.group()
        possessive = re.search(r"['’]s?(?=\s|$)", raw)
        end = m.start() + possessive.end() if possessive else m.end()
        value = _clean_entity(question[m.start():end])
        if value and not _YEAR.search(value) and not re.fullmatch(r"(?:Q[1-4]|USD|CFO|CEO|EPS)", value, re.I):
            names.append((value, m.start(), end))
    return names


def _question_companies(question, kind):
    candidates = _entity_candidates(question)
    # Parse the two alternatives before matching the corpus, so an unknown
    # alternative cannot be silently dropped or mistaken for a similar name.
    if re.search(r"\b(?:which company|which of|compare|between)\b", question, re.I):
        after_colon = question.partition(":")[2]
        paired = _entity_candidates(after_colon) if after_colon else candidates
        if len(paired) >= 2:
            if after_colon:
                return [item[0] for item in paired[:2]]
            for left, right in zip(paired, paired[1:]):
                if re.fullmatch(r"\s+(?:or|versus|vs\.?|and)\s+", question[left[2]:right[1]], re.I):
                    return [left[0], right[0]]
    possessives = [name for name, start, end in candidates
                   if re.search(r"['’](?:s)?(?:\s|$)", question[end-2:end+2], re.I)]
    if kind == "research_report":
        # The company is the target of a report, rather than the named analyst.
        for name, start, _ in reversed(candidates):
            if re.search(r"\b(?:for|of|on|about|gives?|rates?)\s*$", question[:start], re.I):
                return [name]
    if possessives:
        return possessives[:1]
    for name, start, _ in candidates:
        if re.search(r"\b(?:did|does|has|have|audited)\s*$", question[:start], re.I):
            return [name]
    known = [name for name, _, _ in candidates if company_key(name) in _company_names]
    if known:
        return known[:1]
    # This also recognizes a lower-case mention with word boundaries.
    lowered = normalize(question)
    for key in sorted(_company_names, key=len, reverse=True):
        if f" {key} " in lowered:
            return [_company_names[key]]
    return [candidates[0][0]] if candidates else []


def _words(text):
    return normalize(text).strip()


def _tokens(text):
    tokens = []
    for token in _words(text).split():
        if token in _STOP or token.isdigit() or re.fullmatch(r"fy\d{4}[ae]?", token):
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


@dataclass
class _Query:
    parsed: dict
    field: str
    phrase: str
    terms: set[str]
    qualifiers: set[str]
    scoped: bool


def _query_features(question, parsed):
    words = _words(question)
    company_terms = {token for name in parsed["companies"] for token in _tokens(name)}
    terms = set(_tokens(words)) - company_terms
    for field, pattern, phrase in _FIELDS:
        match = re.search(pattern, words)
        if match:
            qualifiers = set(_tokens(words[:match.start()] + words[match.end():])) - company_terms
            # Only financial fields in annual reports require all extra descriptors.
            if parsed["doc_type"] != "annual_report" or field in _PERSON_OR_OFFICE:
                qualifiers = set()
            qualifiers -= {"net", "long", "term", "market", "approximat", "amount", "number", "count", "record"}
            return _Query(parsed, field, phrase, terms, qualifiers,
                          field == "revenue" and bool(qualifiers))
    return _Query(parsed, "", "", terms, terms, False)


def _make_index(chunks):
    groups = defaultdict(list)
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        groups[(company_key(meta.get("company", "")), meta.get("doc_type"))].append(i)
    words = [c.page_content.lower().split() for c in chunks]
    return {"chunks": chunks, "groups": groups, "bm25": BM25Okapi([r or [""] for r in words]) if words else None}


def _in_year(chunk, query):
    year = query.parsed["fiscal_year"]
    doc_year = chunk.metadata.get("fiscal_year")
    if year is None or not isinstance(doc_year, int):
        return True
    kind = query.parsed["doc_type"]
    if kind == "annual_report":
        return year == doc_year or (query.field not in _PERSON_OR_OFFICE and year == doc_year - 1)
    if kind == "earnings_call":
        return doc_year <= year <= doc_year + 1
    return doc_year - 1 <= year <= doc_year + 2


def _evidence_rank(chunk, query):
    """Score the most relevant line, with extra credit for an exact table row."""
    best = 0.0
    supports = False
    body = chunk.page_content
    section = _words(chunk.metadata.get("section", ""))
    for line in body.splitlines():
        row_label, bar, values = line.partition(" | ")
        line_words = _words(line)
        line_tokens = set(_tokens(line))
        extras = query.qualifiers <= line_tokens
        if query.scoped and bar and re.search(r"\brevenue by (?:segment|region)\b", section):
            if query.qualifiers <= set(_tokens(row_label)):
                best = max(best, 14.0)
                supports = True
        if query.phrase:
            exact_row = bar and re.fullmatch(query.phrase, _words(row_label))
            matching = re.search(rf"\b(?:{query.phrase})\b", line_words)
            if query.field in _PERSON_OR_OFFICE and query.phrase and re.search(query.phrase, section):
                # An auditor or head-office section can give the name on its next line.
                matching = matching or len(line_tokens) >= 3
            if exact_row:
                best = max(best, 12.0 if extras else 6.0)
            elif matching and extras:
                best = max(best, 6.0)
            if matching and extras and (bar or query.field in _PERSON_OR_OFFICE
                                        or query.field == "rating" or re.search(r"\d", line)):
                supports = True
            if exact_row and extras:
                supports = True
            if bar and query.field in {"cfo", "ceo"} and matching and "executive officers" in section:
                best = max(best, 13.0)
            if bar and query.field == "employees" and _words(row_label) == "total employees":
                best = max(best, 13.0)
        elif query.terms:
            overlap = len(query.terms & line_tokens) / len(query.terms)
            best = max(best, overlap * 5)
            supports |= overlap == 1 and len(query.terms) >= 2
    year = query.parsed["fiscal_year"]
    if year is not None and year == chunk.metadata.get("fiscal_year"):
        best += 1.5
    if query.field == "auditor" and "independent auditor" in section:
        best += 5
    if query.field == "hq" and re.search(r"(?im)^headquarters\s*:", body):
        best += 5
    if query.parsed["doc_type"] == "research_report" and query.field == "revenue":
        if "estimate changes" in section or ("financial summary" in section and supports):
            best += 5
    return best, supports


def _search(index, question, *, route=True, field_rank=True, prune=True, abstain=True):
    chunks, bm25 = index["chunks"], index["bm25"]
    if not chunks:
        return []
    if not route:
        lexical = bm25.get_scores(question.lower().split())
        order = sorted(range(len(chunks)), key=lambda i: -lexical[i])
        return [chunks[i] for i in order[:MAX_RETRIEVED]]

    query = _query_features(question, parse_query(question))
    keys = [company_key(name) for name in query.parsed["companies"]]
    kind = query.parsed["doc_type"]
    groups = []
    for key in keys:
        ids = index["groups"].get((key, kind), [])
        eligible = [i for i in ids if _in_year(chunks[i], query)]
        if abstain and not eligible:
            return []
        groups.append(eligible or ids or list(range(len(chunks))))
    if not groups:
        if abstain:
            return []
        groups = [list(range(len(chunks)))]

    lexical = bm25.get_scores(question.lower().split())
    scale = max(1.0, max(abs(value) for value in lexical))
    rankings = []
    for ids in groups:
        evidence = {i: _evidence_rank(chunks[i], query) for i in ids} if field_rank else {}
        if field_rank:
            ids = sorted(ids, key=lambda i: -(evidence[i][0] + 2 * lexical[i] / scale))
            supported = [i for i in ids if evidence[i][1]]
            if abstain and not supported:
                return []
            if prune:
                if supported:
                    ids = supported
                # A precise row or statement is sufficient, avoiding competing values.
                ids = ids[:1 if len(groups) > 1 or evidence[ids[0]][0] >= 6 else 2]
        else:
            ids = sorted(ids, key=lambda i: -lexical[i])
        rankings.append(ids)

    selected = []
    seen = set()
    # Round-robin covers both companies before using spare slots for either one.
    for position in range(MAX_RETRIEVED):
        for ranked in rankings:
            if len(selected) == MAX_RETRIEVED:
                return selected
            if position < len(ranked) and ranked[position] not in seen:
                i = ranked[position]
                seen.add(i)
                selected.append(chunks[i])
    return selected


def table_to_text(table_html: str) -> str:
    """Render one HTML table with a caption and the headings on each data row."""
    ### YOUR CODE HERE
    table = BeautifulSoup(table_html, "html.parser").find("table")
    if table is None:
        return ""
    caption = table.find("caption", recursive=False)
    lines = [_text(caption)] if caption else []
    headers = defaultdict(list)
    for tr, row in _table_rows(table):
        cells = list(row.values())
        is_header = tr.find_parent("thead") or (
            not headers and cells and all(c.name == "th" for c in cells) and len({id(c) for c in cells}) > 1
        )
        if is_header:
            for col, cell in row.items():
                label = _text(cell)
                if label and label not in headers[col]:
                    headers[col].append(label)
            continue
        label = _text(row[0]) if 0 in row else ""
        fields = []
        for col, cell in sorted(row.items()):
            if col and cell is not row.get(0):
                header = " / ".join(headers.get(col, ()))
                fields.append(f"{header}: {_text(cell)}" if header else _text(cell))
        if label or fields:
            lines.append(" | ".join([label, *fields]))
    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Create bounded, source-only passages with document and section metadata."""
    ### YOUR CODE HERE
    soup = BeautifulSoup(html, "html.parser")
    metadata = _metadata(file_name, soup)
    chunks = []
    heading = ""
    narrative = []

    def emit(parts, label, kind):
        for block in _sized_blocks(parts, label):
            chunks.append(Document(page_content=block,
                                   metadata={**metadata, "section": label, "block": kind}))

    # Keep table captions and rows together; never splice prose into a table.
    body = soup.body or soup
    for tag in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if tag.find_parent(["table", "nav", "p", "li"]):
            continue
        if tag.name.startswith("h") or tag.name == "table":
            emit(narrative, heading, "prose")
            narrative = []
            if tag.name == "table":
                caption = tag.find("caption", recursive=False)
                label = _text(caption) if caption else heading
                rows = table_to_text(str(tag)).splitlines()
                emit(rows[1:] if caption else rows, label, "table")
            else:
                heading = _text(tag)
        else:
            content = _text(tag)
            if content:
                narrative.append(content)
    emit(narrative, heading, "prose")
    return chunks


def parse_query(question: str) -> dict:
    """Identify the requested companies, year and document type."""
    ### YOUR CODE HERE
    # Determine the requested fiscal year independently from the report year.
    doc_type = _query_type(question)
    year = _YEAR.search(question)
    return {"companies": _question_companies(question, doc_type),
            "fiscal_year": int(year.group(1)) if year else None, "doc_type": doc_type}


def build_index(chunks: list[Document]) -> Any:
    """Build an in-memory BM25 index and map each company to its document types."""
    ### YOUR CODE HERE
    # Keep the actual Document objects so retrieve returns only approved chunks.
    return _make_index(chunks)


def retrieve(index: Any, question: str) -> list[Document]:
    """Select evidence for the question or abstain if it is missing."""
    ### YOUR CODE HERE
    # Require relevant coverage for each named company, including comparisons.
    return _search(index, question)


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
