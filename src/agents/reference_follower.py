import re
from src.ingestion import get_collection
import concurrent.futures

MAX_PER_REF = 3

_NBA_RULE_SEC_RE = re.compile(
    r'\bRule\s+(\d+[A-Z]?)'
    r'[\s,\n]+'
    r'Section\s+([IVXLCDM]+)',
    re.IGNORECASE,
)

_NBA_RULE_RE = re.compile(r'\bRule\s+(\d+[A-Z]?)\b')

_NCAA_RULE_SEC_RE = re.compile(r'\bRule\s+(\d+)-(\d+)', re.IGNORECASE)

_NCAA_RULE_RE = re.compile(r'\bRule\s+(\d+)\b', re.IGNORECASE)

_FIBA_ARTICLE_RE = re.compile(r'\bArticle\s+(\d+)', re.IGNORECASE)

_FIBA_APPENDIX_RE = re.compile(r'\bAppendix\s*\n?\s*([A-Z])\b')

def _and_filter(conditions: list[dict]) -> dict:

    return {"$and": conditions} if len(conditions) > 1 else conditions[0]

def _extract_refs(text: str, league: str) -> list[dict]:

    refs: list[dict] = []
    seen: set[str] = set()

    def _add(where: dict):
        key = str(sorted(str(where).split()))
        if key not in seen:
            seen.add(key)
            refs.append(where)

    league_cond = {"league": {"$eq": league}}

    if league in ("NBA", "WNBA"):

        covered_rules: set[str] = set()
        for m in _NBA_RULE_SEC_RE.finditer(text):
            rn, sn = m.group(1).upper(), m.group(2).upper()
            covered_rules.add(rn)
            _add(_and_filter([league_cond,
                               {"rule_number": {"$eq": rn}},
                               {"section_number": {"$eq": sn}}]))

        for m in _NBA_RULE_RE.finditer(text):
            rn = m.group(1).upper()
            if rn not in covered_rules:
                covered_rules.add(rn)
                _add(_and_filter([league_cond, {"rule_number": {"$eq": rn}}]))

    elif league == "NCAA":
        covered_rules: set[str] = set()

        for m in _NCAA_RULE_SEC_RE.finditer(text):
            rn, sn = m.group(1), m.group(2)
            covered_rules.add(rn)
            _add(_and_filter([league_cond,
                               {"rule_number": {"$eq": rn}},
                               {"section_number": {"$eq": sn}}]))

        for m in _NCAA_RULE_RE.finditer(text):
            rn = m.group(1)

            if re.match(r'\d+\s*-', text[m.end():m.end()+5]):
                continue
            if rn not in covered_rules:
                covered_rules.add(rn)
                _add(_and_filter([league_cond, {"rule_number": {"$eq": rn}}]))

    elif league == "FIBA":
        for m in _FIBA_ARTICLE_RE.finditer(text):
            an = m.group(1)
            _add(_and_filter([league_cond, {"article_number": {"$eq": an}}]))
        for m in _FIBA_APPENDIX_RE.finditer(text):
            al = m.group(1).upper()
            _add(_and_filter([league_cond, {"appendix_letter": {"$eq": al}}]))

    return refs


def follow_references(chunks: list[dict]) -> list[dict]:

    if not chunks:
        return []

    collection = get_collection()
    existing_texts = {c["text"] for c in chunks}
    extra_chunks: list[dict] = []

    queries = []
    seen_queries = set()

    for chunk in chunks:
        league = chunk["metadata"].get("league")
        if not league:
            continue

        for where_filter in _extract_refs(chunk["text"], league):

            key = str(sorted(str(where_filter).split()))
            if key not in seen_queries:
                seen_queries.add(key)
                queries.append(where_filter)

    if not queries:
        return []

    def fetch_ref(where_filter):
        try:
            return collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
                limit=MAX_PER_REF,
            )
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for results in executor.map(fetch_ref, queries):
            if results and results.get("documents"):
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    if doc not in existing_texts:
                        existing_texts.add(doc)
                        extra_chunks.append({
                            "text": doc,
                            "metadata": meta,
                            "distance": None,
                            "rerank_score": None,
                            "followed_ref": True,
                        })

    return extra_chunks
