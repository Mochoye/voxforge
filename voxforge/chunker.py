import pysbd

# One segmenter instance is enough since it's stateless and thread-safe
_segmenter = pysbd.Segmenter(language="en", clean=True)

# Tuning constants — adjust based on what you observe during testing
MIN_CHUNK_CHARS = 80     # chunks shorter than this get merged with the next
MAX_CHUNK_CHARS = 250    # chunks longer than this get split at a comma/semicolon


def _split_long_sentence(sentence: str) -> list[str]:
    """
    If a single sentence exceeds MAX_CHUNK_CHARS, split it at the last
    comma or semicolon before the midpoint. This prevents feeding the
    model extremely long sequences.
    """
    if len(sentence) <= MAX_CHUNK_CHARS:
        return [sentence]

    # Find split candidates: commas and semicolons
    midpoint = len(sentence) // 2
    best_split = -1

    # Search around the midpoint for a natural break
    for i in range(midpoint, max(0, midpoint - 80), -1):
        if sentence[i] in (",", ";"):
            best_split = i + 1
            break

    if best_split == -1:
        # No good split found — just return as-is, model will handle it
        return [sentence]

    left = sentence[:best_split].strip()
    right = sentence[best_split:].strip()

    # Recursively split if still too long
    return _split_long_sentence(left) + _split_long_sentence(right)


def chunk(text: str) -> list[str]:
    """
    Split normalized text into synthesis-ready chunks.

    Method:
      1. Use pysbd to detect sentence boundaries
      2. Split any sentence over MAX_CHUNK_CHARS at punctuation
      3. Merge consecutive short sentences under MIN_CHUNK_CHARS

    Input:  a normalized string (output of normalizer.normalize())
    Output: list of strings, each safe to pass to the TTS model
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    # Step 1: Sentence boundary detection
    sentences = _segmenter.segment(text)

    # Step 2: Split long sentences
    split_sentences = []
    for s in sentences:
        split_sentences.extend(_split_long_sentence(s))

    # Step 3: Merge short consecutive chunks
    merged: list[str] = []
    buffer = ""

    for sentence in split_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if buffer:
            candidate = buffer + " " + sentence
            if len(buffer) < MIN_CHUNK_CHARS:
                # Buffer is short — merge current sentence into it
                buffer = candidate
            else:
                # Buffer is long enough — flush it and start new buffer
                merged.append(buffer)
                buffer = sentence
        else:
            buffer = sentence

    # Flush remaining buffer
    if buffer:
        merged.append(buffer)

    return merged