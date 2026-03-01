from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from models.analytics_events import DetectedKeyword
from services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS: list[str] = [
    "fire",
    "gun",
    "gunshot",
    "shot",
    "stabbing",
    "stabbed",
    "bleeding",
    "blood",
    "chest pain",
    "not breathing",
    "cant breathe",
    "can't breathe",
    "unconscious",
    "unresponsive",
    "overdose",
    "seizure",
    "heart attack",
    "stroke",
    "choking",
    "drowning",
    "trapped",
    "bomb",
    "explosion",
    "active shooter",
]


def _load_keywords(path: str | None) -> list[str]:
    """Load keywords from a JSON file, falling back to defaults."""
    if path:
        p = Path(path)
        if p.is_file():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list) and all(isinstance(k, str) for k in data):
                    return data
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load keywords from %s, using defaults", p)
    return list(DEFAULT_KEYWORDS)


class KeywordSpotter:
    """Text-based keyword spotting on the ASR transcript stream.

    Subscribes to the ``EventEmitter`` and performs case-insensitive
    regex matching on every ``TranscriptEvent`` against a configurable
    list of emergency dispatch keywords.

    Parameters
    ----------
    keywords : list[str] | None
        Explicit keyword list. If ``None``, loaded from *keywords_path*
        or falls back to ``DEFAULT_KEYWORDS``.
    keywords_path : str | None
        Path to a JSON file containing a flat list of keyword strings.
    cooldown_s : float
        Minimum seconds between repeated detections of the same keyword.
    """

    def __init__(
        self,
        keywords: list[str] | None = None,
        keywords_path: str | None = None,
        cooldown_s: float = 5.0,
    ) -> None:
        kw_list = keywords if keywords is not None else _load_keywords(keywords_path)
        self._keywords = kw_list
        self._cooldown_s = cooldown_s

        escaped = [re.escape(k) for k in kw_list]
        self._pattern = re.compile(
            r"\b(?:" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )

    async def run(
        self,
        call_id: str,
        emitter: EventEmitter,
        hit_queue: asyncio.Queue[DetectedKeyword],
    ) -> None:
        """Subscribe to *emitter* and match transcripts against keywords."""
        eq = await emitter.subscribe(call_id)
        last_fired: dict[str, float] = {}

        logger.info(
            "KeywordSpotter started for call %s (%d keywords)",
            call_id,
            len(self._keywords),
        )

        try:
            while True:
                raw = await eq.get()
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                if data.get("event") != "transcript":
                    continue

                text = data.get("original_transcript", "")
                seq = data.get("seq")
                now = time.monotonic()

                for match in self._pattern.finditer(text):
                    keyword = match.group().lower()
                    prev = last_fired.get(keyword, 0.0)
                    if now - prev < self._cooldown_s:
                        continue
                    last_fired[keyword] = now

                    kw = DetectedKeyword(
                        keyword=keyword,
                        transcript_seq=seq,
                    )
                    logger.info(
                        "KWS detection: '%s' in transcript seq=%s on call %s",
                        keyword,
                        seq,
                        call_id,
                    )
                    try:
                        hit_queue.put_nowait(kw)
                    except asyncio.QueueFull:
                        pass
        except asyncio.CancelledError:
            return
        finally:
            await emitter.unsubscribe(call_id, eq)
            logger.info("KeywordSpotter finished for call %s", call_id)
