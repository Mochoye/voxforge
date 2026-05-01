import shelve
import os
import torch
import hashlib
from pathlib import Path


CACHE_PATH = "models/speaker_cache"   # shelve adds .db extension automatically


class SpeakerCache:
    """
    Persistent cache for speaker embeddings using Python shelve.

    Keys   : SHA256 hash of the reference audio file
    Values : dict with gpt_cond_latent, speaker_embedding tensors + metadata

    Survives process restarts. Located at models/speaker_cache.db
    Phase 4 will replace this with Redis for multi-process access.
    """

    def __init__(self, cache_path: str = CACHE_PATH):
        self.cache_path = cache_path
        # Ensure the directory exists
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"[SpeakerCache] Cache path: {cache_path}.db")

    def _open(self):
        return shelve.open(self.cache_path)

    def get(self, audio_hash: str, device: str = "cuda") -> dict | None:
        """
        Retrieve cached speaker embedding by audio hash.

        Returns None if not in cache.
        Returns dict with gpt_cond_latent and speaker_embedding on correct device.
        """
        with self._open() as db:
            if audio_hash not in db:
                return None

            entry = db[audio_hash]

            # Move tensors to the correct device
            return {
                "gpt_cond_latent": entry["gpt_cond_latent"].to(device),
                "speaker_embedding": entry["speaker_embedding"].to(device),
                "source_file": entry.get("source_file", "unknown"),
                "duration": entry.get("duration", 0.0),
                "voice_ratio": entry.get("voice_ratio", 0.0),
            }

    def set(
        self,
        audio_hash: str,
        gpt_cond_latent: torch.Tensor,
        speaker_embedding: torch.Tensor,
        metadata: dict = None,
    ):
        """
        Store a speaker embedding in the cache.
        Tensors are moved to CPU before storing to avoid device issues on reload.
        """
        metadata = metadata or {}

        with self._open() as db:
            db[audio_hash] = {
                "gpt_cond_latent": gpt_cond_latent.cpu(),
                "speaker_embedding": speaker_embedding.cpu(),
                "source_file": metadata.get("source_file", ""),
                "duration": metadata.get("duration", 0.0),
                "voice_ratio": metadata.get("voice_ratio", 0.0),
            }

        print(f"[SpeakerCache] Stored embedding for hash: {audio_hash[:12]}...")

    def has(self, audio_hash: str) -> bool:
        """Check if a hash exists in the cache without loading tensors."""
        with self._open() as db:
            return audio_hash in db

    def delete(self, audio_hash: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        with self._open() as db:
            if audio_hash in db:
                del db[audio_hash]
                return True
        return False

    def clear(self):
        """Wipe the entire cache. Useful for testing."""
        with self._open() as db:
            db.clear()
        print(f"[SpeakerCache] Cache cleared.")

    def list_entries(self) -> list[dict]:
        """List all cached entries with metadata (no tensors)."""
        entries = []
        with self._open() as db:
            for key in db:
                entry = db[key]
                entries.append({
                    "hash": key[:12] + "...",
                    "source_file": entry.get("source_file", "unknown"),
                    "duration": entry.get("duration", 0.0),
                    "voice_ratio": entry.get("voice_ratio", 0.0),
                })
        return entries

    def size(self) -> int:
        """Number of cached embeddings."""
        with self._open() as db:
            return len(db)