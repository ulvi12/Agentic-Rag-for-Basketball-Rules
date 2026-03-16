def split_large_chunk_with_overlap(text: str, max_chars: int = 1500, overlap: int = 150) -> list:

    if len(text) <= max_chars:
        return [text]

    sub_chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + max_chars

        if end >= text_length:
            sub_chunks.append(text[start:].strip())
            break

        chunk_slice = text[start:end]
        break_point = max(chunk_slice.rfind('\n'), chunk_slice.rfind(' '))

        if break_point != -1 and break_point > overlap:
            end = start + break_point

        sub_chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in sub_chunks if c]