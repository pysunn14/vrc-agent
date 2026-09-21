from __future__ import annotations

import unicodedata


FAST_STOP_TRANSCRIPT = "멈춰"


def normalize_stop_transcript(transcript: str) -> str:
    if not isinstance(transcript, str):
        raise TypeError("transcript must be a string")
    normalized = unicodedata.normalize("NFKC", transcript)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def is_fast_stop(transcript: str) -> bool:
    return normalize_stop_transcript(transcript) == FAST_STOP_TRANSCRIPT
