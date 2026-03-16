import os
import re
import time
from openai import OpenAI
from typing import Optional

from src.retriever import retrieve_per_league
from src.agents.reranker import rerank
from src.agents.reference_follower import follow_references
from src.agents.decomposer import decompose
from src.agents.classifier import classify

_openai_client = None
def _get_llm_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", "missing_key"),
        )
    return _openai_client

MODEL_NAME = "openai/gpt-5-mini"

def _build_prompt(question: str, grouped_sources: list[dict]) -> str:

    context_parts = []
    for i, src in enumerate(grouped_sources, start=1):
        label = src["label"]
        combined_text = "\n...\n".join(src["texts"])
        context_parts.append(f"[Source {i} — {label}]\n{combined_text}")

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""You are a basketball rules expert. Answer using ONLY the numbered sources below.
Rules:
- Be brief where possible. No fluff, no preamble.
- Cite sources inline using ONLY square-bracket numbers: [N] matching the source label exactly.
- NEVER write (Source: ...) or (NBA Rule ...) or any other citation format — only [N].
- Do not invent sub-numbers (e.g. never write [6.2] or [3.1]).
- If the context does not contain enough information, say so in one sentence.
- Never disregard a league, if the question asks about it.
- Answer each asked leagued separately with headers for each league (when more than one league
  is asked).
- When answering for each league, analyze the definitions, synonyms, different contexts and more.

CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER (inline citations as [N] only):"""

def _format_source_label(meta: dict) -> str:

    parts = [meta.get("league", "Unknown")]
    if "rule_number" in meta:
        parts.append(f"Rule {meta['rule_number']}")
    if "rule_name" in meta:
        parts.append(meta["rule_name"])
    if "section_number" in meta:
        parts.append(f"Section {meta['section_number']}")
    if "section_name" in meta:
        parts.append(meta["section_name"])
    if "article_number" in meta:
        parts.append(f"Article {meta['article_number']}")
    if "appendix_letter" in meta:
        parts.append(f"Appendix {meta['appendix_letter']}")
    if "appendix_roman" in meta:
        parts.append(f"Appendix {meta['appendix_roman']}")
    if "appendix_name" in meta:
        parts.append(meta["appendix_name"])
    return " | ".join(parts)

def answer(
    question: str,
    league: Optional[str] = None,
    history: Optional[list[dict]] = None,
    trace: bool = False,
) -> dict:

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    question_type = classify(question, history=history)
    timings["classify"] = round(time.perf_counter() - t0, 2)

    if question_type == "retry":
        result = {
            "answer": "That's the best answer I can provide based on the official rulebooks. The sources are listed above.",
            "sources": [],
        }
        if trace:
            result["trace"] = {"question_type": "retry", "timings": timings}
        return result

    if question_type == "conversational":
        messages = list(history or [])
        messages.append({"role": "user", "content": question})
        t0 = time.perf_counter()
        client = _get_llm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        timings["generate"] = round(time.perf_counter() - t0, 2)
        raw = response.choices[0].message.content
        answer_text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        result = {"answer": answer_text, "sources": []}
        if trace:
            result["trace"] = {"question_type": "conversational", "timings": timings}
        return result

    if question_type == "out_of_scope":
        result = {
            "answer": "This question doesn't appear to be about basketball rules.",
            "sources": [],
        }
        if trace:
            result["trace"] = {"question_type": "out_of_scope", "timings": timings}
        return result

    t0 = time.perf_counter()
    sub_questions = decompose(question, history=history)
    timings["decompose"] = round(time.perf_counter() - t0, 2)

    all_chunks: list[dict] = []
    seen_texts: set[str] = set()
    all_leagues: list[str] = []
    t_retrieve = t_rerank = t_ref_follow = 0.0

    for sub_item in sub_questions:
        sub_q = sub_item["query"]
        if league:
            leagues = [league.upper()]
        else:
            leagues = sub_item["leagues"]

        all_leagues.extend(l for l in leagues if l not in all_leagues)

        if not leagues:
            continue

        t0 = time.perf_counter()
        chunks = retrieve_per_league(sub_q, leagues=leagues)
        t_retrieve += time.perf_counter() - t0

        t0 = time.perf_counter()
        extra = follow_references(chunks)
        t_ref_follow += time.perf_counter() - t0
        seen_ref = {c["text"] for c in chunks}
        for c in extra:
            if c["text"] not in seen_ref:
                seen_ref.add(c["text"])
                chunks.append(c)

        t0 = time.perf_counter()
        chunks = rerank(sub_q, chunks)
        t_rerank += time.perf_counter() - t0

        for c in chunks:
            if c["text"] not in seen_texts:
                seen_texts.add(c["text"])
                all_chunks.append(c)

    if t_retrieve:   timings["retrieve"]  = round(t_retrieve, 2)
    if t_ref_follow: timings["ref_follow"] = round(t_ref_follow, 2)
    if t_rerank:     timings["rerank"]    = round(t_rerank, 2)

    if not all_chunks:
        return {"answer": "I couldn't find any relevant content in the rulebooks for this question.", "sources": []}

    grouped_sources = []
    seen_labels = {}
    for c in all_chunks:
        label = _format_source_label(c["metadata"])
        if label not in seen_labels:
            seen_labels[label] = {
                "label": label,
                "metadata": c["metadata"],
                "texts": []
            }
            grouped_sources.append(seen_labels[label])
        seen_labels[label]["texts"].append(c["text"])

    messages = list(history or [])
    messages.append({"role": "user", "content": _build_prompt(question, grouped_sources)})

    t0 = time.perf_counter()
    client = _get_llm_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    timings["generate"] = round(time.perf_counter() - t0, 2)

    raw = response.choices[0].message.content
    answer_text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    cited_numbers = []
    for match in re.finditer(r'\[(\d+)\]', answer_text):
        num = int(match.group(1))
        if num not in cited_numbers and 1 <= num <= len(grouped_sources):
            cited_numbers.append(num)

    used_sources = []
    old_to_new = {}
    for new_id, old_id in enumerate(cited_numbers, start=1):
        used_sources.append(grouped_sources[old_id - 1])
        old_to_new[old_id] = new_id

    def _renumber_citation(match):
        old_id = int(match.group(1))
        if old_id in old_to_new:
            return f"[{old_to_new[old_id]}]"
        return ""

    answer_text = re.sub(r'\[(\d+)\]', _renumber_citation, answer_text)

    result = {
        "answer": answer_text,
        "sources": [src["metadata"] for src in used_sources],
        "all_sources": [src["metadata"] for src in grouped_sources],
    }

    if trace:
        timings["total"] = round(sum(timings.values()), 2)
        reranked = [c for c in all_chunks if not c.get("followed_ref")]
        result["trace"] = {
            "question_type": "lookup",
            "sub_questions": [sq["query"] for sq in sub_questions],
            "leagues_queried": all_leagues,
            "chunks_after_rerank": len(reranked),
            "followed_refs_added": len(all_chunks) - len(reranked),
            "timings": timings,
        }

    return result