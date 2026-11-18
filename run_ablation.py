"""Measure each retrieval change using the public corpus and the provided scorer."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

import hw4


@dataclass(frozen=True)
class Variant:
    name: str
    change: str
    tables: bool = False
    sections: bool = False
    route: bool = False
    rank: bool = False
    prune: bool = False
    abstain: bool = False


VARIANTS = (
    Variant("B0", "Plain text, recursive splits, whole-corpus BM25"),
    Variant("V1", "Preserve table captions and label every cell", tables=True),
    Variant("V2", "Split at section and table boundaries", tables=True, sections=True),
    Variant("V3", "Route by company, year and document type", tables=True, sections=True, route=True),
    Variant("V4", "Rank by row labels and other evidence in the passage",
            tables=True, sections=True, route=True, rank=True),
    Variant("V5", "Keep only the strongest passages",
            tables=True, sections=True, route=True, rank=True, prune=True),
    Variant("V6", "Abstain when eligible evidence is missing",
            tables=True, sections=True, route=True, rank=True, prune=True, abstain=True),
)
RESULT_START, RESULT_END = "<!-- hw4-results-start -->", "<!-- hw4-results-end -->"


def collect(variant, corpus, questions, gold, docs, tables):
    """Build fresh chunks and record the runner's evidence checks before generation."""
    started = time.perf_counter()
    hw4._company_names.clear()
    if variant.sections:
        grouped = {name: hw4.chunk_document(name, html) for name, html in corpus.items()}
    else:
        grouped = {name: hw4._split_plain(name, html, variant.tables) for name, html in corpus.items()}
    chunks = [chunk for group in grouped.values() for chunk in group]
    permitted = {chunk.page_content for chunk in chunks}
    index = hw4.build_index(chunks)
    table_scores = []
    for table in tables:
        html = table["html"]
        text = hw4.table_to_text(html) if variant.tables else BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        table_scores.append(hw4.table_score(text, table))
    docs_pass = sum(not hw4.document_problems(grouped.get(f), None, m, f) for f, m in docs.items())
    rows, evidence = {}, {}
    for qid, question in questions.items():
        if variant.abstain:
            retrieved = hw4.retrieve(index, question)
        else:
            retrieved = hw4._search(index, question, route=variant.route, field_rank=variant.rank,
                                    prune=variant.prune, abstain=False)
        valid = len(retrieved) <= hw4.MAX_RETRIEVED and all(c.page_content in permitted for c in retrieved)
        if not valid:
            raise ValueError(f"{variant.name} returned an invalid chunk list for {qid}")
        parsed = hw4.parse_query(question) if variant.route else {}
        expected = gold[qid]
        rows[qid] = {
            "variant": variant.name, "id": qid, "category": expected["category"], "question": question,
            "parsed": json.dumps(parsed), "parse_score": round(hw4.parse_score(parsed, expected), 3),
            "retrieved": ";".join(sorted({c.metadata["source"] for c in retrieved})),
            "valid": valid, "retrieval_hit": hw4.retrieval_hit(retrieved, expected) if expected["evidence"] else "",
            "answer": "", "correct": "", "expected": "",
        }
        evidence[qid] = retrieved
    return rows, evidence, {
        "variant": variant.name, "table_cells": sum(table_scores) / len(table_scores),
        "documents_pass": docs_pass, "document_count": len(docs), "chunks": len(chunks),
        "retrieval_seconds": round(time.perf_counter() - started, 3),
    }


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No data for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def verify_runner_csv(path, final_rows):
    """Reuse fresh official-run answers only if their retrieval records match V6."""
    with path.open(newline="", encoding="utf-8") as stream:
        original = list(csv.DictReader(stream))
    if len(original) != len(final_rows) or {row["id"] for row in original} != set(final_rows):
        raise ValueError("results.csv and the experiment use different questions")
    for row in original:
        ref = final_rows[row["id"]]
        for key in ("question", "category", "retrieved", "valid", "retrieval_hit", "parsed"):
            if row[key] != str(ref[key]):
                raise ValueError(f"results.csv disagrees with V6 on {row['id']} ({key}). Run hw4.py again.")
        if not row["answer"].strip():
            raise ValueError(f"Generator returned an empty response for {row['id']}. Run hw4.py again.")
    return {row["id"]: row["answer"] for row in original}


def generate_answers(variant, questions, evidence, cache, workers):
    """Use the supplied generator, sharing an answer only for an identical complete prompt."""
    answers, pending = {}, {}
    for qid, question in questions.items():
        chunks = evidence[qid]
        prompt_key = (question, hw4.format_context(chunks))
        if not chunks:
            answers[qid] = hw4.generate(question, [])
        elif prompt_key in cache:
            answers[qid] = cache[prompt_key]
        else:
            pending[qid] = prompt_key
    print(f"{variant}: {len(pending)} new generator calls", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(hw4.generate, questions[qid], evidence[qid]): qid for qid in pending}
        for future in as_completed(futures):
            qid = futures[future]
            try:
                answer = future.result().strip()
                if not answer:
                    raise ValueError("empty response")
            except Exception as exc:
                raise RuntimeError(f"{variant}: generator failed on {qid} ({type(exc).__name__})") from None
            cache[pending[qid]] = answers[qid] = answer
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                print(f"  {variant}: {completed}/{len(pending)} complete", flush=True)
    return answers


def summarize(rows, metrics, answers, gold):
    if answers is not None:
        for qid, row in rows.items():
            correct = hw4.answer_correct(answers[qid], gold[qid])
            row.update(answer=answers[qid], correct=correct, expected="" if correct else gold[qid]["answer"])
    answerable = [row for qid, row in rows.items() if gold[qid]["evidence"]]
    unanswerable = [row for qid, row in rows.items() if not gold[qid]["evidence"]]
    hits = sum(bool(row["retrieval_hit"]) for row in answerable)
    kept = sum(bool(row["retrieved"]) for row in answerable)
    rejected = sum(not row["retrieved"] for row in unanswerable)
    correct = "" if answers is None else sum(row["correct"] for row in rows.values())
    return {**metrics, "recall_hits": hits, "answerable": len(answerable), "kept": kept,
            "rejected": rejected, "unanswerable": len(unanswerable), "correct": correct,
            "question_count": len(rows)}


def _rate(count, n):
    return f"{count}/{n} ({count / n:.1%})" if n else "0/0"


def _inline(text):
    return str(text).replace("|", "\\|").replace("\n", " ").replace("`", "'")


def report_section(metrics, all_rows, offline):
    lines = [
        "| Variant | One change from the previous row | Recall@5 | Kept | Rejected | Answer accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    rows_by_variant = {name: {r["id"]: r for r in all_rows if r["variant"] == name}
                       for name in (v.name for v in VARIANTS)}
    for variant, m in zip(VARIANTS, metrics):
        n, u = m["answerable"], m["unanswerable"]
        accuracy = "Pending API run" if offline else _rate(m["correct"], m["question_count"])
        lines.append(f"| {variant.name} | {variant.change} | {_rate(m['recall_hits'], n)} | "
                     f"{m['kept']}/{n} | {m['rejected']}/{u} | {accuracy} |")
    first, last = metrics[0], metrics[-1]
    lines += [
        "",
        f"All variants use the same {last['question_count']} public questions and the same BM25 settings. "
        f"Recall@5 requires every evidence item for a question. Kept counts answerable questions with evidence "
        f"returned, and rejected counts unanswerable questions with an empty result. V6 preserved "
        f"{last['table_cells']:.1%} of scored table cells and passed {last['documents_pass']}/{last['document_count']} "
        f"document checks. B0 produced {first['chunks']} chunks and V6 produced {last['chunks']}.",
        "",
        ("These retrieval measurements ran locally. Answer accuracy will be filled in after the provided "
         "DeepSeek generator runs on the retrieved passages." if offline else
         "Answer accuracy comes from the provided DeepSeek generator and grading functions. Identical questions "
         "with identical formatted contexts share a response. The detailed rows are in "
         "[ablation_results.csv](ablation_results.csv). The official final run is in [results.csv](results.csv)."),
        "",
        "#### What each change did",
        "",
    ]
    interpretation = (
        "Row labels and units remained attached to the corresponding values.",
        "Section boundaries separated table rows from company names on the cover. Global BM25 still needed metadata to recover that connection.",
        "Company and report-year filters narrowed the search, and comparison questions received passages from both companies.",
        "The score favoured a matching row or statement over a passage with only scattered query words.",
        "Shorter contexts kept the most specific evidence and made fewer competing columns and figures visible to the generator.",
        "The evidence check handled unavailable companies, years and metrics before calling the generator.",
    )
    for previous, current, explanation in zip(metrics, metrics[1:], interpretation):
        earlier, later = previous["recall_hits"], current["recall_hits"]
        if earlier == later:
            rec = f"retrieval hits stayed at {later}"
        else:
            verb = "rose" if later > earlier else "fell"
            rec = f"retrieval hits {verb} from {earlier} to {later}"
        accuracy = ("" if offline else f" Correct answers went from {previous['correct']} to {current['correct']}.")
        lines.append(f"- **{current['variant']}** {explanation} {rec.capitalize()}.{accuracy}")
        if current["variant"] == "V6":
            lines[-1] += f" It rejected {current['rejected']} of {current['unanswerable']} unanswerable questions."
    lines += ["", "#### Error analysis", ""]
    baseline_rows = rows_by_variant["B0"]
    final = rows_by_variant["V6"]
    repaired = [r for r in final.values() if r["retrieval_hit"] is True and baseline_rows[r["id"]]["retrieval_hit"] is False]
    comparison = next((r for r in repaired if r["category"] == "comparison"), None)
    for row in [r for r in (comparison, next((r for r in repaired if r is not comparison), None)) if r]:
        label = "at least one required fact" if row["category"] == "comparison" else "the required fact"
        lines.append(f"- **{row['id']}** asks “{_inline(row['question'])}” B0 missed {label}. "
                     f"V6 found the evidence in `{row['retrieved']}`.")
    retrieval_misses = [r for r in final.values() if r["retrieval_hit"] is False]
    for row in retrieval_misses[:4]:
        lines.append(f"- **{row['id']}** remained a retrieval miss. The question asks “{_inline(row['question'])}” "
                     f"V6 returned `{row['retrieved'] or 'no passages'}`.")
    if not offline:
        wrong = [row for row in final.values() if not row["correct"]]
        for row in wrong[:8]:
            detail = ("The passage contains the scored fact, so the error is in reading or using the context."
                      if row["retrieval_hit"] is True else "The retrieved context did not meet the evidence requirement.")
            lines.append(f"- **{row['id']}** produced `{_inline(row['answer'])}` instead of "
                         f"`{_inline(row['expected'])}`. {detail} Question: {_inline(row['question'])}")
        if not wrong:
            lines.append("- V6 answered all public questions correctly. The misses above show where the baseline lost evidence.")
    return "\n".join(lines)


def update_readme(path, section):
    text = path.read_text(encoding="utf-8")
    before, start, rest = text.partition(RESULT_START)
    _, end, after = rest.partition(RESULT_END)
    if not start or not end:
        raise ValueError(f"The ablation report markers are missing in {path}")
    path.write_text(before + start + "\n" + section + "\n" + end + after, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=Path("public_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("ablation"))
    parser.add_argument("--offline", action="store_true", help="Measure retrieval metrics without calling DeepSeek")
    parser.add_argument("--final-results", type=Path, help="Reuse V6 answers from a fresh python hw4.py run")
    parser.add_argument("--write-readme", type=Path, help="Fill the marked results section of this README")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.offline and not os.getenv("DEEPSEEK_API_KEY"):
        parser.error("Set DEEPSEEK_API_KEY in the environment or in .env before running the generator")
    corpus = hw4.load_corpus(args.data_folder / "corpus")
    questions, gold, docs, tables = [
        hw4.load_json(args.data_folder / name)
        for name in ("questions.json", "ground_truth.json", "docs.json", "tables.json")
    ]
    if not corpus or not all((questions, gold, docs, tables)) or set(questions) != set(gold):
        parser.error("This study needs a labelled corpus such as public_test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measured = {}
    for variant in VARIANTS:
        print(f"Building {variant.name}: {variant.change}", flush=True)
        measured[variant.name] = collect(variant, corpus, questions, gold, docs, tables)
    cache = {}
    if args.final_results and not args.offline:
        final_rows, final_evidence, _ = measured[VARIANTS[-1].name]
        final_answers = verify_runner_csv(args.final_results, final_rows)
        for qid, answer in final_answers.items():
            cache[questions[qid], hw4.format_context(final_evidence[qid])] = answer
    metrics, all_rows = [], []
    for variant in VARIANTS:
        rows, evidence, info = measured[variant.name]
        answers = None if args.offline else generate_answers(variant.name, questions, evidence, cache, args.workers)
        summary = summarize(rows, info, answers, gold)
        metrics.append(summary)
        all_rows.extend(rows.values())
        print(f"{variant.name}: recall {summary['recall_hits']}/{summary['answerable']}, "
              f"rejected {summary['rejected']}/{summary['unanswerable']}, "
              f"accuracy {'pending' if answers is None else str(summary['correct']) + '/' + str(len(rows))}", flush=True)
    write_csv(args.output_dir / "ablation_results.csv", all_rows)
    write_csv(args.output_dir / "ablation_metrics.csv", metrics)
    if args.write_readme:
        update_readme(args.write_readme, report_section(metrics, all_rows, args.offline))
    print(f"Generator usage for this script: {hw4.GENERATOR_USAGE.summary()}", flush=True)


if __name__ == "__main__":
    main()
