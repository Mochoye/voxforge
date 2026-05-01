import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxforge.speaker_cache import SpeakerCache

TEST_CACHE_PATH = "models/test_cache_temp"


def get_test_cache():
    cache = SpeakerCache(cache_path=TEST_CACHE_PATH)
    cache.clear()
    return cache


def make_dummy_tensors():
    latent = torch.zeros(1, 1, 1024)
    embedding = torch.zeros(1, 512, 1)
    return latent, embedding


def test_store_and_retrieve():
    cache = get_test_cache()
    latent, embedding = make_dummy_tensors()

    cache.set("hash_abc123", latent, embedding,
              metadata={"source_file": "test.wav", "duration": 10.0, "voice_ratio": 0.8})

    result = cache.get("hash_abc123", device="cpu")
    assert result is not None, "Should retrieve stored embedding"
    assert result["source_file"] == "test.wav"
    assert result["duration"] == 10.0
    print("  PASS: store and retrieve")


def test_cache_miss():
    cache = get_test_cache()
    result = cache.get("nonexistent_hash", device="cpu")
    assert result is None, "Should return None for missing key"
    print("  PASS: cache miss returns None")


def test_has():
    cache = get_test_cache()
    latent, embedding = make_dummy_tensors()
    cache.set("hash_xyz", latent, embedding)
    assert cache.has("hash_xyz") is True
    assert cache.has("hash_nothere") is False
    print("  PASS: has() works correctly")


def test_delete():
    cache = get_test_cache()
    latent, embedding = make_dummy_tensors()
    cache.set("hash_to_delete", latent, embedding)
    assert cache.has("hash_to_delete") is True
    deleted = cache.delete("hash_to_delete")
    assert deleted is True
    assert cache.has("hash_to_delete") is False
    print("  PASS: delete works")


def test_size():
    cache = get_test_cache()
    latent, embedding = make_dummy_tensors()
    cache.set("h1", latent, embedding)
    cache.set("h2", latent, embedding)
    assert cache.size() == 2
    print("  PASS: size() returns correct count")


def test_clear():
    cache = get_test_cache()
    latent, embedding = make_dummy_tensors()
    cache.set("h1", latent, embedding)
    cache.clear()
    assert cache.size() == 0
    print("  PASS: clear() wipes cache")


def test_cpu_storage():
    """Tensors should always be stored on CPU regardless of input device."""
    cache = get_test_cache()
    latent, embedding = make_dummy_tensors()
    cache.set("hash_cpu_test", latent, embedding)
    result = cache.get("hash_cpu_test", device="cpu")
    assert result["gpt_cond_latent"].device.type == "cpu"
    print("  PASS: tensors stored on CPU")


if __name__ == "__main__":
    print("Running speaker cache tests...\n")
    test_store_and_retrieve()
    test_cache_miss()
    test_has()
    test_delete()
    test_size()
    test_clear()
    test_cpu_storage()
    print("\nAll speaker cache tests passed.")