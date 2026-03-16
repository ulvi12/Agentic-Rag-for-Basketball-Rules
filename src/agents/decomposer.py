import re
import json
import os
from openai import OpenAI
from typing import Optional

_openai_client = None
def _get_llm_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", "missing_key"),
        )
    return _openai_client

_DECOMPOSE_PROMPT = """\
Your task is to prepare search queries for a basketball rules knowledge base.
Available leagues: NBA, WNBA, NCAA, FIBA. (misspellings are possible, e.g. fiba, nba, wnba, ncaa, etc.)
Given the conversation history and the latest question, return a JSON array of
objects, where each object has a "query" (a self-contained search query) and
"leagues" (a list of applicable leagues for that query).

Rules:
- If the grammar, spelling, or punctuation is broken, fix it to a proper sentence.
- If the question names specific leagues, include only those in "leagues".
- If the question is general without specifying any leagues (e.g. "what is a flagrant foul?"), include all four leagues in "leagues".
- If the question refers to a previous topic (e.g. "what about in WNBA?"), REWRITE it 
as a fully self-contained query using the history (e.g. "What are the WNBA foul rules?").
- If the question compares multiple leagues, split into one query per league, keeping each sub-query neutral and symmetric (do not inject your own knowledge or add specifics to one league that weren't in the original question).
- If the question asks about multiple distinct topics, split into one query per topic.
- Maximum 8 sub-queries, but use them up only when it is necessary (e.g. question asks about two topics in all four leagues.).

Return ONLY a valid JSON array of objects. No explanation, no markdown.

Examples:
  History: Q: "How many personal fouls before disqualification in NBA?"
  Question: "What about in WNBA?"
  → [{"query": "How many personal fouls before disqualification in WNBA?", "leagues": ["WNBA"]}]

  History: (none)
  Question: "How do nba and fiba shot clock rules differ?"
  → [{"query": "What are the NBA shot clock rules?", "leagues": ["NBA"]}, {"query": "What are the FIBA shot clock rules?", "leagues": ["FIBA"]}]

  History: (none)
  Question: "How many fouls before disqualification?"
  → [{"query": "How many fouls before disqualification?", "leagues": ["NBA", "WNBA", "NCAA", "FIBA"]}]
"""

ALL_LEAGUES = ["NBA", "WNBA", "NCAA", "FIBA"]

def decompose(
    question: str,
    history: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> list[dict]:

    from src.rag import MODEL_NAME
    _model = model or MODEL_NAME

    default_fallback = [{"query": question, "leagues": ALL_LEAGUES}]

    try:
        messages = [{"role": "system", "content": _DECOMPOSE_PROMPT}]
        if history:
            messages.extend(history[-10:])
        messages.append({"role": "user", "content": question})

        client = _get_llm_client()
        response = client.chat.completions.create(
            model=_model,
            messages=messages,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return default_fallback

        parsed = json.loads(match.group())
        valid_subs = []
        for item in parsed:
            if isinstance(item, dict) and "query" in item and "leagues" in item:
                valid_leagues = [l.upper() for l in item["leagues"] if l.upper() in ALL_LEAGUES]
                if not valid_leagues:
                    valid_leagues = ALL_LEAGUES
                valid_subs.append({"query": str(item["query"]).strip(), "leagues": valid_leagues})

        return valid_subs if valid_subs else default_fallback

    except Exception:
        return default_fallback