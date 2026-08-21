# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


class _TerminatingMatcher:
    """Matcher stub that rejects calls after its terminating token."""

    def __init__(self, terminating_token: int):
        self.terminating_token = terminating_token
        self.accepted_tokens: list[int] = []
        self.terminated = False
        self.rollback_count = 0

    def accept_token(self, token: int) -> bool:
        assert not self.terminated, "matcher called after grammar termination"
        self.accepted_tokens.append(token)
        self.terminated = token == self.terminating_token
        return True

    def is_terminated(self) -> bool:
        return self.terminated

    def rollback(self, num_tokens: int) -> None:
        self.rollback_count = num_tokens
        self.terminated = False

    def reset(self) -> None:
        self.accepted_tokens.clear()
        self.terminated = False
        self.rollback_count = 0


def test_validate_tokens_stops_at_grammar_termination():
    """Speculative validation trims every token after the stop token."""
    matcher = _TerminatingMatcher(terminating_token=2)
    grammar = XgrammarGrammar(vocab_size=8, matcher=matcher, ctx=Mock())

    accepted = grammar.validate_tokens([1, 2, 3, 4])

    assert accepted == [1, 2]
    assert matcher.accepted_tokens == [1, 2]
    assert matcher.rollback_count == 2
    assert not matcher.is_terminated()


def test_validate_tokens_skips_an_already_terminated_grammar():
    """A committed terminal grammar has no valid speculative suffix."""
    matcher = _TerminatingMatcher(terminating_token=2)
    grammar = XgrammarGrammar(
        vocab_size=8,
        matcher=matcher,
        ctx=Mock(),
        _is_terminated=True,
    )

    assert grammar.validate_tokens([1, 2]) == []
    assert matcher.accepted_tokens == []


def test_accept_tokens_stops_at_grammar_termination():
    """Committed token batches do not advance beyond grammar termination."""
    matcher = _TerminatingMatcher(terminating_token=2)
    grammar = XgrammarGrammar(vocab_size=8, matcher=matcher, ctx=Mock())

    assert grammar.accept_tokens("request", [1, 2, 3, 4])
    assert grammar.is_terminated()
    assert grammar.num_processed_tokens == 2
    assert matcher.accepted_tokens == [1, 2]


def test_accept_tokens_after_termination_is_a_noop():
    """Scheduler re-entry after termination preserves the terminal state."""
    matcher = _TerminatingMatcher(terminating_token=2)
    grammar = XgrammarGrammar(vocab_size=8, matcher=matcher, ctx=Mock())
    assert grammar.accept_tokens("request", [1, 2])

    assert grammar.accept_tokens("request", [3, 4])
    assert grammar.num_processed_tokens == 2
    assert matcher.accepted_tokens == [1, 2]


def test_reset_clears_matcher_and_grammar_state():
    """A reset grammar accepts a new sequence from its initial state."""
    matcher = _TerminatingMatcher(terminating_token=2)
    grammar = XgrammarGrammar(vocab_size=8, matcher=matcher, ctx=Mock())
    assert grammar.accept_tokens("request", [1, 2])

    grammar.reset()

    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0
    assert matcher.accepted_tokens == []
    assert grammar.accept_tokens("request", [1])
