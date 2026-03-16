import re
import fitz
from typing import Optional, Literal
from pydantic import BaseModel
from .utils import split_large_chunk_with_overlap

PDF_PATH = "Rulebooks/NCAA.pdf"

TEXT_SKIP_START = 23995
TEXT_SKIP_END = -21766

class NCAAMetadata(BaseModel):
    league: Literal["NCAA"] = "NCAA"
    category: Literal["front_matter", "rule", "appendix"]
    rule_number: Optional[str] = None
    rule_name: Optional[str] = None
    section_number: Optional[str] = None
    section_name: Optional[str] = None
    appendix_roman: Optional[str] = None
    appendix_name: Optional[str] = None

def _extract_text(pdf_path: str = PDF_PATH) -> str:
    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:

            extracted = page.get_text()
            if extracted:
                full_text += extracted + "\n"
    return full_text[TEXT_SKIP_START:TEXT_SKIP_END]

def _parse_rulebook(raw_text: str) -> list[dict]:

    cleaned = re.sub(r'\\s*', '', raw_text)

    cleaned = re.sub(r'^\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)

    cleaned = re.sub(r'\n?\s*\d*\s*\n?RULE \d+\s*/\s*[^\n]+\s*\n?\d*\s*\n?', '\n', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'\n?\s*\d*\s*\n?Appendix [IVXLCDM]+\s*/\s*[^\n]+\s*\n?\d*\s*\n?', '\n', cleaned)

    cleaned = re.sub(r'Art\.\s*\n\s*(\d)', r'Art. \1', cleaned)

    cleaned = re.sub(r' {2,}', '. ', cleaned)

    app_split = re.split(r'(?=\nAppendix I\b)', cleaned, maxsplit=1)
    appendixes = app_split[1] if len(app_split) > 1 else ""
    front_and_rules = app_split[0]

    rule_split = re.split(r'(?=\nRULE 1\b)', front_and_rules, maxsplit=1)
    front_matter = rule_split[0] if len(rule_split) > 1 else ""
    main_rules = rule_split[1] if len(rule_split) > 1 else rule_split[0]

    final_chunks = []

    if front_matter and "RULE 1" in cleaned:
        final_chunks.append({"category": "front_matter", "text": front_matter.strip()})

    if main_rules and "RULE" in main_rules:
        for rule in re.split(r'(?=\nRULE \d+\b)', '\n' + main_rules.strip()):
            rule = rule.strip()
            if not rule.startswith("RULE"):
                continue
            section_chunks = re.split(r'(?=\nSECTION \d+\s*\.)', '\n' + rule)
            rule_intro = section_chunks[0].strip()
            if len(section_chunks) > 1:
                for section in section_chunks[1:]:
                    if section.strip():
                        final_chunks.append({"category": "rule", "text": f"{rule_intro}\n{section.strip()}"})
            else:
                if rule_intro:
                    final_chunks.append({"category": "rule", "text": rule_intro})

    if appendixes:
        for app in re.split(r'(?=\nAppendix [IVXLCDM]+\b)', '\n' + appendixes.strip()):
            if app.strip():
                final_chunks.append({"category": "appendix", "text": app.strip()})

    return final_chunks

def _extract_metadata(chunk: dict) -> dict:
    text, category = chunk["text"], chunk["category"]
    metadata = NCAAMetadata(category=category)

    if category == "rule":
        m = re.search(r'RULE (\d+)\s*\n([^\n]+)', text)
        if m:
            metadata.rule_number = m.group(1).strip()
            metadata.rule_name = m.group(2).strip()

        m = re.search(r'SECTION (\d+)\s*\.\s*(.*?)(?=\s*\n\s*Art\.)', text, re.DOTALL)
        if m:
            metadata.section_number = m.group(1).strip()
            metadata.section_name = re.sub(r'\s+', ' ', m.group(2)).strip()
        else:
            m = re.search(r'SECTION (\d+)\s*\.\s*([^\n]+)', text)
            if m:
                metadata.section_number = m.group(1).strip()
                metadata.section_name = m.group(2).strip()

    elif category == "appendix":
        m = re.search(r'Appendix\s+([IVXLCDM]+)\s*[\n—]+\s*([^\n]+)', text)
        if m:
            metadata.appendix_roman = m.group(1).strip()
            metadata.appendix_name = m.group(2).strip()

    return metadata.model_dump(exclude_none=True)

def load_ncaa_documents(pdf_path: str = PDF_PATH, max_chars: int = 1500, overlap: int = 200) -> list[dict]:

    print(f"[NCAA] Extracting text from {pdf_path}...")
    raw_text = _extract_text(pdf_path)
    chunks = _parse_rulebook(raw_text)
    print(f"[NCAA] Parsed {len(chunks)} structural chunks.")

    documents = []
    for chunk in chunks:
        meta = _extract_metadata(chunk)
        for sub in split_large_chunk_with_overlap(chunk["text"], max_chars, overlap):
            documents.append({"page_content": sub, "metadata": meta.copy()})

    print(f"[NCAA] Produced {len(documents)} overlapping documents for DB.")
    return documents