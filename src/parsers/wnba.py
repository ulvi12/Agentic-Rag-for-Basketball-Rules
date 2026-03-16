import re
import PyPDF2
from typing import Optional, Literal
from pydantic import BaseModel
from .utils import split_large_chunk_with_overlap

PDF_PATH = "Rulebooks/WNBA.pdf"

TEXT_SKIP_START = 5210

class WNBAMetadata(BaseModel):
    league: Literal["WNBA"] = "WNBA"
    category: Literal["rule", "appendix"]
    rule_number: Optional[str] = None
    rule_name: Optional[str] = None
    section_number: Optional[str] = None
    section_name: Optional[str] = None
    appendix_letter: Optional[str] = None
    appendix_name: Optional[str] = None

def _extract_text(pdf_path: str = PDF_PATH) -> str:
    full_text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
    return full_text[TEXT_SKIP_START:]

def _parse_rulebook(raw_text: str) -> list[dict]:
    cleaned = re.sub(r'\s*-\s*\d+\s*-\s*', '\n', raw_text)

    parts = re.split(r'COMMENTS ON THE RULES', cleaned, maxsplit=1)
    main_rules = parts[0]
    appendix = "COMMENTS ON THE RULES\n" + parts[1] if len(parts) > 1 else ""

    final_chunks = []

    for rule in re.split(r'(?=RULE NO\. \d+\s*—)', main_rules):
        if not rule.strip():
            continue
        section_chunks = re.split(r'(?=Section [IVXLCDM]+\s*—)', rule)
        rule_intro = section_chunks[0].strip()
        if len(section_chunks) > 1:
            for section in section_chunks[1:]:
                if section.strip():
                    final_chunks.append({"category": "rule", "text": f"{rule_intro}\n{section.strip()}"})
        else:
            if rule_intro:
                final_chunks.append({"category": "rule", "text": rule_intro})

    if appendix:
        letter_chunks = re.split(r'\n(?=[A-Z]\.\s+[A-Z])', appendix)
        for lc in letter_chunks:
            if lc.strip():
                final_chunks.append({"category": "appendix", "text": lc.strip()})

    final_chunks = final_chunks[0:-19] + final_chunks[-18:]
    return final_chunks

def _extract_metadata(chunk: dict) -> dict:
    text, category = chunk["text"], chunk["category"]
    metadata = WNBAMetadata(category=category)

    if category == "rule":
        m = re.search(r'RULE NO\. (\d+[A-Z]?)\s*—\s*([^\n]+)', text)
        if m:
            metadata.rule_number = m.group(1).strip()
            metadata.rule_name = m.group(2).strip()

        m = re.search(r'Section ([IVXLCDM]+)\s*—\s*([^\n]+)', text)
        if m:
            metadata.section_number = m.group(1).strip()
            metadata.section_name = m.group(2).strip()

    elif category == "appendix":
        m = re.search(r'^([A-Z])\.\s+([^\n]+)', text.strip())
        if m:
            metadata.appendix_letter = m.group(1).strip()
            metadata.appendix_name = m.group(2).strip()

    return metadata.model_dump(exclude_none=True)

def load_wnba_documents(pdf_path: str = PDF_PATH, max_chars: int = 1500, overlap: int = 200) -> list[dict]:

    print(f"[WNBA] Extracting text from {pdf_path}...")
    raw_text = _extract_text(pdf_path)
    chunks = _parse_rulebook(raw_text)
    print(f"[WNBA] Parsed {len(chunks)} structural chunks.")

    documents = []
    for chunk in chunks:
        meta = _extract_metadata(chunk)
        for sub in split_large_chunk_with_overlap(chunk["text"], max_chars, overlap):
            documents.append({"page_content": sub, "metadata": meta.copy()})

    print(f"[WNBA] Produced {len(documents)} overlapping documents for DB.")
    return documents