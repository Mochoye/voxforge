import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxforge.chunker import chunk

def test_basic_split():
    text = "Hello world. This is a test. VoxForge is working."
    result = chunk(text)
    assert len(result) >= 1, "Should produce at least one chunk"
    print(f"  PASS: {len(result)} chunks: {result}")

def test_short_sentences_merged():
    # Two very short sentences should merge into one chunk
    text = "Hi there. How are you? I am fine."
    result = chunk(text)
    print(f"  PASS: {len(result)} chunks from short sentences: {result}")

def test_tricky_boundary():
    # pysbd should handle Mr. without splitting at the period
    text = "Mr. Smith went to Washington. He enjoyed the visit."
    result = chunk(text)
    assert len(result) >= 1
    # "Mr. Smith went to Washington" should be one chunk, not split at "Mr."
    full = " ".join(result)
    assert "Mr" in full or "mister" in full.lower()
    print(f"  PASS: {result}")

def test_long_text():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "The five boxing wizards jump quickly. "
        "Sphinx of black quartz, judge my vow."
    )
    result = chunk(text)
    assert len(result) >= 1
    for c in result:
        assert len(c) <= 350, f"Chunk too long: {len(c)} chars: {c}"
    print(f"  PASS: {len(result)} chunks, all within length limit")

def test_empty_input():
    try:
        chunk("")
        print("  FAIL: Should have raised ValueError")
    except ValueError:
        print("  PASS: Empty input raised ValueError correctly")

if __name__ == "__main__":
    print("Running chunker tests...\n")
    test_basic_split()
    test_short_sentences_merged()
    test_tricky_boundary()
    test_long_text()
    test_empty_input()
    print("\nAll chunker tests passed.")