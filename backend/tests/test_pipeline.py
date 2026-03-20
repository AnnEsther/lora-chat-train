"""backend/tests/test_pipeline.py — Unit tests for the data pipeline modules."""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://lora:lora@localhost:5432/lora")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")


# ── TranscriptExtractor ───────────────────────────────────────────────────────

class TestTranscriptExtractor:
    def _ext(self):
        from training.extractor.transcript_extractor import TranscriptExtractor
        return TranscriptExtractor()

    def test_extracts_two_pairs_from_two_turns(self):
        transcript = [
            {"role": "user",      "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user",      "content": "And what is the capital of Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."},
        ]
        candidates = self._ext().extract(transcript)
        assert len(candidates) == 2
        assert "Paris" in candidates[0].assistant_turn
        assert "Berlin" in candidates[1].assistant_turn

    def test_skips_sleep_command(self):
        transcript = [
            {"role": "user",      "content": "/sleep"},
            {"role": "assistant", "content": "Going to sleep now."},
        ]
        assert len(self._ext().extract(transcript)) == 0

    def test_redacts_api_keys(self):
        from training.extractor.transcript_extractor import TranscriptExtractor
        cleaned = TranscriptExtractor()._clean("Here is my key: sk-abcdefghijklmnopqrstu")
        assert "sk-abcdefghijklmnopqrstu" not in cleaned
        assert "[REDACTED_API_KEY]" in cleaned

    def test_redacts_email_addresses(self):
        from training.extractor.transcript_extractor import TranscriptExtractor
        cleaned = TranscriptExtractor()._clean("Contact me at user@example.com please.")
        assert "user@example.com" not in cleaned
        assert "[REDACTED_EMAIL]" in cleaned

    def test_skips_turns_that_are_too_short(self):
        transcript = [
            {"role": "user",      "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert len(self._ext().extract(transcript)) == 0

    def test_skips_first_user_turn_when_followed_by_another_user(self):
        # user → user → assistant: only the second user/assistant pair is valid
        transcript = [
            {"role": "user",      "content": "First long enough user message here please."},
            {"role": "user",      "content": "Second long enough user message here please."},
            {"role": "assistant", "content": "This is a response to the second user message."},
        ]
        candidates = self._ext().extract(transcript)
        assert len(candidates) == 1
        assert candidates[0].user_turn == "Second long enough user message here please."

    def test_skips_system_role_turns(self):
        transcript = [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "Tell me something interesting about astronomy."},
            {"role": "assistant", "content": "The observable universe is about 93 billion light-years in diameter."},
        ]
        candidates = self._ext().extract(transcript)
        assert len(candidates) == 1

    def test_single_unanswered_user_turn_produces_no_candidate(self):
        transcript = [
            {"role": "user", "content": "This message has no assistant reply at all."},
        ]
        assert len(self._ext().extract(transcript)) == 0


# ── Curator ───────────────────────────────────────────────────────────────────

class TestCurator:
    def _curator(self):
        from training.curator.curator import Curator
        return Curator()

    def _c(self, user, asst, _id="fake-id"):
        return {"user_turn": user, "assistant_turn": asst, "_id": _id}

    def test_high_quality_candidate_is_included(self):
        c = self._c(
            "How do I sort a list in Python?",
            "You can sort a list using `list.sort()` or `sorted(list)`. "
            "For example: `numbers = [3, 1, 2]; numbers.sort()` gives `[1, 2, 3]`.",
        )
        result = self._curator().score_and_filter([c])[0]
        assert result["included"] is True
        assert result["score"] > 0.5

    def test_very_short_answer_scores_low(self):
        c = self._c("What is the Pythagorean theorem?", "a^2+b^2=c^2")
        result = self._curator().score_and_filter([c])[0]
        assert result["score"] < 0.9

    def test_redacted_content_is_excluded(self):
        c = self._c("Here is my config.", "[REDACTED_API_KEY] found in your turn.")
        result = self._curator().score_and_filter([c])[0]
        assert result["included"] is False
        assert "unsafe_pattern" in (result["rejection_reason"] or "")

    def test_refusal_response_is_excluded(self):
        c = self._c(
            "Can you help me with something?",
            "I can't help with that request. I am unable to assist you.",
        )
        result = self._curator().score_and_filter([c])[0]
        assert result["included"] is False

    def test_multiple_candidates_scored_independently(self):
        candidates = [
            self._c(
                "How does a hash map work?",
                "A hash map stores key-value pairs using a hash function to map keys to "
                "array indices, providing O(1) average-case lookup and insertion.",
                _id="id-1",
            ),
            self._c("Hello", "[REDACTED_EMAIL] contact info.", _id="id-2"),
        ]
        results = self._curator().score_and_filter(candidates)
        assert results[0]["included"] is True
        assert results[1]["included"] is False


# ── DatasetWriter ─────────────────────────────────────────────────────────────

class TestDatasetWriter:
    def test_writes_valid_jsonl(self):
        import json
        from training.datasets.dataset_writer import DatasetWriter

        jsonl = DatasetWriter().write_jsonl([
            {"user_turn": "What is 2+2?",    "assistant_turn": "It is 4."},
            {"user_turn": "What is the sun?", "assistant_turn": "A star at the centre of our solar system."},
        ])
        lines = [l for l in jsonl.splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert [m["role"] for m in obj["messages"]] == ["system", "user", "assistant"]

    def test_system_prompt_present_in_every_example(self):
        import json
        from training.datasets.dataset_writer import DatasetWriter

        jsonl = DatasetWriter(system_prompt="You are a test assistant.").write_jsonl([
            {"user_turn": "Ping", "assistant_turn": "Pong " * 10},
        ])
        obj = json.loads(jsonl)
        assert obj["messages"][0]["content"] == "You are a test assistant."

    def test_write_to_file(self, tmp_path):
        from training.datasets.dataset_writer import DatasetWriter

        path = DatasetWriter().write_to_file(
            [{"user_turn": "Hello there.", "assistant_turn": "Hi! How can I help you today?"}],
            tmp_path / "dataset.jsonl",
        )
        assert path.exists()
        assert path.stat().st_size > 0
