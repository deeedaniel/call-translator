from __future__ import annotations

import pytest

from models.asr_events import TranscriptEvent
from services.punctuated_buffer import PunctuatedBufferStreamer


def _transcript(
    text: str,
    *,
    call_id: str = "call-1",
    seq: int = 0,
    lang: str = "es",
    is_final: bool = False,
) -> TranscriptEvent:
    return TranscriptEvent(
        call_id=call_id,
        seq=seq,
        original_transcript=text,
        detected_language=lang,
        asr_confidence=0.95,
        is_final=is_final,
    )


@pytest.fixture
def streamer() -> PunctuatedBufferStreamer:
    return PunctuatedBufferStreamer()


class TestAbsoluteTranscriptDedup:
    """Verify that absolute (accumulated) ASR transcripts don't duplicate."""

    def test_no_duplication_across_partials(self, streamer: PunctuatedBufferStreamer):
        chunks = streamer.feed(_transcript("The patient"))
        assert chunks == []

        chunks = streamer.feed(_transcript("The patient has"))
        assert chunks == []

        chunks = streamer.feed(
            _transcript("The patient has chest pain.", is_final=True)
        )
        assert len(chunks) == 1
        assert chunks[0].text == "The patient has chest pain."

    def test_mid_utterance_punctuation(self, streamer: PunctuatedBufferStreamer):
        chunks = streamer.feed(_transcript("The patient has a fever, and"))
        assert len(chunks) == 1
        assert chunks[0].text == "The patient has a fever,"

        chunks = streamer.feed(
            _transcript("The patient has a fever, and chest pain.", is_final=True)
        )
        assert len(chunks) == 1
        assert chunks[0].text == "and chest pain."


class TestClauseBoundarySplitting:
    def test_single_sentence_with_period_final(self, streamer: PunctuatedBufferStreamer):
        chunks = streamer.feed(
            _transcript("Help me please.", is_final=True)
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Help me please."
        assert chunks[0].source_language == "es"
        assert chunks[0].call_id == "call-1"

    def test_multi_clause_with_commas(self, streamer: PunctuatedBufferStreamer):
        chunks = streamer.feed(
            _transcript("First clause, second clause, third clause.", is_final=True)
        )
        assert len(chunks) == 3
        assert chunks[0].text == "First clause,"
        assert chunks[1].text == "second clause,"
        assert chunks[2].text == "third clause."

    def test_question_and_exclamation(self, streamer: PunctuatedBufferStreamer):
        chunks = streamer.feed(
            _transcript("Is it serious? Yes! Very.", is_final=True)
        )
        assert len(chunks) == 3
        assert chunks[0].text == "Is it serious?"
        assert chunks[1].text == "Yes!"
        assert chunks[2].text == "Very."

    def test_semicolon_boundary(self, streamer: PunctuatedBufferStreamer):
        chunks = streamer.feed(
            _transcript("Pain in chest; shortness of breath.", is_final=True)
        )
        assert len(chunks) == 2
        assert chunks[0].text == "Pain in chest;"
        assert chunks[1].text == "shortness of breath."


class TestPartialBuffering:
    def test_no_punctuation_waits_for_final(self, streamer: PunctuatedBufferStreamer):
        chunks = streamer.feed(_transcript("The patient has"))
        assert chunks == []

        chunks = streamer.feed(_transcript("The patient has chest"))
        assert chunks == []

        chunks = streamer.feed(
            _transcript("The patient has chest pain", is_final=True)
        )
        assert len(chunks) == 1
        assert chunks[0].text == "The patient has chest pain"

    def test_cursor_resets_between_utterances(self, streamer: PunctuatedBufferStreamer):
        streamer.feed(
            _transcript("First utterance.", seq=0, is_final=True)
        )
        chunks = streamer.feed(
            _transcript("Second utterance.", seq=1, is_final=True)
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Second utterance."


class TestMultipleCalls:
    def test_independent_call_ids(self, streamer: PunctuatedBufferStreamer):
        chunks_a = streamer.feed(_transcript("Hello, world.", call_id="A", is_final=True))
        chunks_b = streamer.feed(_transcript("Hola, mundo.", call_id="B", is_final=True))

        assert len(chunks_a) == 2
        assert chunks_a[0].call_id == "A"

        assert len(chunks_b) == 2
        assert chunks_b[0].call_id == "B"


class TestFlush:
    def test_flush_clears_state(self, streamer: PunctuatedBufferStreamer):
        streamer.feed(_transcript("buffered text"))
        streamer.flush("call-1")
        # After flush, new feed should start fresh
        chunks = streamer.feed(
            _transcript("New text.", is_final=True)
        )
        assert len(chunks) == 1
        assert chunks[0].text == "New text."
