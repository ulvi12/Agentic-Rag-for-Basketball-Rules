import re
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

_CLASSIFY_PROMPT = """\
You are classifying a user question for a basketball rules assistant. You MUST choose the category you think is most likely.

Return ONLY one word from these four options:
- "lookup"         — the user asks specifically about basketball rules, definitions, or regulations (NOT scores, stats, or player performance)
- "conversational" — the user is genuinely asking a follow-up about a basketball rule question that needs explanation \
(e.g. "can you clarify?", "give me an example", "what does that mean?", "explain")
- "retry"          — the user is asking to redo, verify, or challenge the previous answer about basketball rules \
(e.g. "are you sure?", "try again", "try harder", "are you certain?", "double check", \
"that doesn't seem right", "can you do better?")
- "out_of_scope"   — the question is not about basketball rules

Examples:
  "How many fouls before disqualification?" -> "lookup"
  "What is a flagrant foul?" -> "lookup"
  "can you clarify that?" -> "conversational"
  "give me an example" -> "conversational"
  "what about overtime?" -> "lookup"
  "are you sure?" -> "retry"
  "try again" -> "retry"
  "that doesn't look right" -> "retry"
  "are you certain?" -> "retry"
  "who is lebron james?" -> "out_of_scope"
  "what is the weather like?" -> "out_of_scope"
  "how many points did lebron score last night?" -> "out_of_scope"
"""

def classify(
    question: str,
    history: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> str:
    from src.rag import MODEL_NAME
    _model = model or MODEL_NAME

    try:
        messages = [{"role": "system", "content": _CLASSIFY_PROMPT}]
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
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().lower()
        if "retry" in raw:
            return "retry"
        if "conversational" in raw:
            return "conversational"
        if "out_of_scope" in raw:
            return "out_of_scope"
        return "lookup"

    except Exception:
        return "lookup"