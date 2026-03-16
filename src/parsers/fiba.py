import re
import fitz
from typing import Optional, Literal
from pydantic import BaseModel
from .utils import split_large_chunk_with_overlap

PDF_PATH = "Rulebooks/FIBA.pdf"

TEXT_SKIP_START = 11832
TEXT_SKIP_END = -50897

class FIBAMetadata(BaseModel):
    league: Literal["FIBA"] = "FIBA"
    category: Literal["rule", "appendix"]
    rule_number: Optional[str] = None
    rule_name: Optional[str] = None
    article_number: Optional[str] = None
    article_name: Optional[str] = None
    appendix_letter: Optional[str] = None
    appendix_name: Optional[str] = None

def _extract_text(pdf_path: str = PDF_PATH) -> str:
    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            full_text += page.get_text() + "\n"
    return full_text[TEXT_SKIP_START:TEXT_SKIP_END]

def _parse_rulebook(raw_text: str) -> list[dict]:

    cleaned = re.sub(r'\n?Page \d+ of \d+\s*\nOFFICIAL BASKETBALL RULES \d+\s*\n[A-Za-z]+\s+\d+\n?', '\n', raw_text)
    cleaned = re.sub(r'\n?[A-Za-z]+\s+\d+\s*\nOFFICIAL BASKETBALL RULES \d+\s*\nPage \d+ of \d+\n?', '\n', cleaned)

    parts = re.split(r'(?=\nAPPENDIX A\b)', '\n' + cleaned, maxsplit=1)
    main_rules = parts[0]
    appendixes = parts[1] if len(parts) > 1 else ""

    final_chunks = []

    for rule in re.split(r'(?=\nRULE [A-Z]+\s*–)', main_rules):
        if not rule.strip() or "RULE" not in rule:
            continue
        article_chunks = re.split(r'(?=\nArticle \d+\s*\n)', '\n' + rule.strip())
        rule_intro = article_chunks[0].strip()
        if len(article_chunks) > 1:
            for article in article_chunks[1:]:
                if article.strip():
                    final_chunks.append({"category": "rule", "text": f"{rule_intro}\n{article.strip()}"})
        else:
            if rule_intro:
                final_chunks.append({"category": "rule", "text": rule_intro})

    if appendixes:
        for app in re.split(r'(?=\nAPPENDIX [A-Z]\b)', '\n' + appendixes.strip()):
            if app.strip():
                final_chunks.append({"category": "appendix", "text": app.strip()})

    return final_chunks

def _extract_metadata(chunk: dict) -> dict:
    text, category = chunk["text"], chunk["category"]
    metadata = FIBAMetadata(category=category)

    if category == "rule":
        m = re.search(r'^RULE ([A-Z]+)\s*–\s*([^\n]+)', text)
        if m:
            metadata.rule_number = m.group(1).strip()
            metadata.rule_name = m.group(2).strip()

        m = re.search(r'\nArticle (\d+)\s*\n([^\n]+)', '\n' + text)
        if m:
            metadata.article_number = m.group(1).strip()
            metadata.article_name = m.group(2).strip()

    elif category == "appendix":
        m = re.search(r'^APPENDIX ([A-Z])\s*–\s*([^\n]+)', text)
        if m:
            metadata.appendix_letter = m.group(1).strip()
            metadata.appendix_name = m.group(2).strip()

    return metadata.model_dump(exclude_none=True)

def load_fiba_documents(pdf_path: str = PDF_PATH, max_chars: int = 1500, overlap: int = 200) -> list[dict]:

    print(f"Extracting text from {pdf_path}")
    raw_text = _extract_text(pdf_path)
    chunks = _parse_rulebook(raw_text)
    print(f"Parsed {len(chunks)} structural chunks.")

    documents = []
    for chunk in chunks:
        meta = _extract_metadata(chunk)
        for sub in split_large_chunk_with_overlap(chunk["text"], max_chars, overlap):
            documents.append({"page_content": sub, "metadata": meta.copy()})

    print(f"Produced {len(documents)} overlapping documents for DB.")
    return documents