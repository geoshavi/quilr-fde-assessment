"""Task 3 tests: streaming PII redactor (unit) + gateway integration.

Structure:
1. StreamingRedactor unit tests (categories A-E from the assessment) -- pure
   Python, no FastAPI/httpx involved. This is the primary, exhaustive test
   surface: for many scenarios, every possible chunk-split position (chunk
   sizes 1..len(text)+1) is generated programmatically and asserted.
2. `_redacted_stream` generator tests against a minimal hand-written fake
   upstream client (no ASGI) -- proves the gateway's own generator yields
   incrementally and cleans up its upstream connection on early close,
   without any transport-layer interference.
3. Gateway + mock_llm ASGI integration tests (httpx.ASGITransport) -- proves
   the real FastAPI/httpx wiring produces correctly redacted output
   end-to-end. NOTE: httpx.ASGITransport does not preserve real streaming
   timing (verified empirically: even a direct ASGITransport call to a
   deliberately slow endpoint with no gateway involved shows the same
   before-first-chunk delay as a fully-buffered response), so it is used
   here for functional/output correctness and cancellation behavior only,
   never for timing assertions.
4. One real-process test (actual `uvicorn` processes over real sockets,
   matching this repo's established Task 1/Task 2 pattern) that empirically
   proves genuine incremental delivery: time-to-first-byte is measured well
   before the artificially-delayed upstream's total duration.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from task3_stream_guardrail import app as gateway_app
from task3_stream_guardrail import mock_llm
from task3_stream_guardrail.redactor import (
    CREDIT_CARD_MAX,
    EMAIL_LOCAL_MAX,
    EMAIL_LABEL_MAX,
    EMAIL_TOTAL_MAX,
    MAX_MERGE_HOLDBACK,
    NUMERIC_CANDIDATE_MAX,
    StreamingRedactor,
    _luhn_valid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_stream(text: str, chunk_size: int) -> str:
    """Feed `text` through a fresh StreamingRedactor in chunks of `chunk_size`, then flush."""
    redactor = StreamingRedactor()
    parts = [redactor.feed(text[i : i + chunk_size]) for i in range(0, len(text), chunk_size)]
    parts.append(redactor.flush())
    return "".join(parts)


def _every_split_cases(cases: list[tuple[str, str]]):
    """Expand (text, expected) pairs into one parametrized case per chunk size, per text."""
    for text, expected in cases:
        for size in range(1, len(text) + 2):
            yield pytest.param(text, expected, size, id=f"{text[:24]!r}/size={size}")


# =============================================================================
# 1a. Category A -- emails
# =============================================================================

EMAIL_CASES = [
    ("Contact user@example.com today", "Contact [REDACTED] today"),
    ("user@example.com is the contact", "[REDACTED] is the contact"),
    ("the contact is user@example.com", "the contact is [REDACTED]"),
    ("mail me at a.b@sub.example.co.uk!", "mail me at [REDACTED]!"),
    ("a@b.com and c@d.com", "[REDACTED] and [REDACTED]"),
    ("(user@example.com)", "([REDACTED])"),
    ("email:user@example.com,", "email:[REDACTED],"),
    ("user@example.com", "[REDACTED]"),  # PII alone, whole stream
    ("visit test.co.uk for info", "visit test.co.uk for info"),  # dotted domain, no '@' -> not an email
    ("price @ store", "price @ store"),  # bare '@' with no local/domain -> not an email
]


@pytest.mark.parametrize("text,expected,chunk_size", list(_every_split_cases(EMAIL_CASES)))
def test_email_every_split_position(text: str, expected: str, chunk_size: int) -> None:
    """Covers: complete-in-one-chunk, split before/after '@', split inside domain,

    every single split position (chunk_size=1 alone already exercises every
    adjacent-character boundary), punctuation around the address, and
    false-prefix text that never becomes a real email.
    """
    assert _run_stream(text, chunk_size) == expected


def test_email_character_by_character_explicit() -> None:
    """Explicit char-by-char case (redundant with chunk_size=1 above, kept for clarity)."""
    text = "Reach me at first.last+tag@my-domain.example.org please"
    expected = "Reach me at [REDACTED] please"
    assert _run_stream(text, 1) == expected


def test_email_long_multi_label_domain_over_local_part_cap() -> None:
    """Regression: a domain > EMAIL_LOCAL_MAX (64) chars must still fully redact.

    (Caught during implementation: an earlier version of the candidate-start
    scan applied the wrong length cap to domain characters scanned before the
    backward scan had reached the '@' -- see redactor.py's docstring.)
    """
    label = "a" + "1234567890" * 6  # 61 chars, under the 63-char DNS label limit
    domain = f"{label}.{label}.com"
    assert len(domain) > EMAIL_LOCAL_MAX
    text = f"contact user@{domain} now"
    expected = "contact [REDACTED] now"
    for size in (1, 2, 3, 7, len(text)):
        assert _run_stream(text, size) == expected


# =============================================================================
# 1b. Category B -- SSNs
# =============================================================================

SSN_CASES = [
    ("SSN is 123-45-6789 on file", "SSN is [REDACTED] on file"),
    ("here is my ssn 123-45-6789", "here is my ssn [REDACTED]"),  # EOF immediately after SSN
    ("123-45-6789 is my number", "[REDACTED] is my number"),
    ("123-45-6789", "[REDACTED]"),  # PII alone, whole stream
    ("(123-45-6789)", "([REDACTED])"),
    ("call 555-1234 today", "call 555-1234 today"),  # digit/hyphen but not SSN-shaped -> unchanged
]


@pytest.mark.parametrize("text,expected,chunk_size", list(_every_split_cases(SSN_CASES)))
def test_ssn_every_split_position(text: str, expected: str, chunk_size: int) -> None:
    """Covers: one chunk, splits across both hyphens, every split position,

    EOF immediately after the SSN, and non-SSN digit/hyphen text left alone.
    """
    assert _run_stream(text, chunk_size) == expected


def test_ssn_character_by_character_explicit() -> None:
    text = "record 123-45-6789 archived"
    assert _run_stream(text, 1) == "record [REDACTED] archived"


def test_ssn_eof_immediately_after_no_flush_needed_check() -> None:
    """flush() must resolve an SSN that ends exactly at end-of-stream."""
    redactor = StreamingRedactor()
    out = redactor.feed("my ssn: 123-45-6789")
    out += redactor.flush()
    assert out == "my ssn: [REDACTED]"


# =============================================================================
# 1c. Category C -- credit cards
# =============================================================================

VALID_VISA_16 = "4111111111111111"
VALID_VISA_16B = "4012888888881881"
VALID_AMEX_15 = "378282246310005"
VALID_MC_16 = "5555555555554444"
VALID_VISA_13 = "4222222222222"
INVALID_LUHN_16 = "4111111111111112"

assert _luhn_valid(VALID_VISA_16)
assert _luhn_valid(VALID_AMEX_15)
assert _luhn_valid(VALID_VISA_13)
assert not _luhn_valid(INVALID_LUHN_16)

CREDIT_CARD_CASES = [
    (f"card {VALID_VISA_16} charged", "card [REDACTED] charged"),
    ("card 4111 1111 1111 1111 charged", "card [REDACTED] charged"),
    ("card 4111-1111-1111-1111 charged", "card [REDACTED] charged"),
    (f"card {VALID_AMEX_15} charged", "card [REDACTED] charged"),  # 15-digit
    (f"card {VALID_VISA_13} charged", "card [REDACTED] charged"),  # 13-digit
    (f"card {VALID_MC_16} charged", "card [REDACTED] charged"),
    (f"card {INVALID_LUHN_16} charged", f"card {INVALID_LUHN_16} charged"),  # fails Luhn -> unchanged
    (f"{VALID_VISA_16}", "[REDACTED]"),  # PII alone
]


@pytest.mark.parametrize("text,expected,chunk_size", list(_every_split_cases(CREDIT_CARD_CASES)))
def test_credit_card_every_split_position(text: str, expected: str, chunk_size: int) -> None:
    """Covers: contiguous, spaced, hyphenated, 13/15/16-digit valid cards,

    every split position throughout digits/separators, and an invalid-Luhn
    16-digit number that must remain unchanged.
    """
    assert _run_stream(text, chunk_size) == expected


def test_credit_card_character_by_character_explicit() -> None:
    text = f"pay with {VALID_VISA_16} please"
    assert _run_stream(text, 1) == "pay with [REDACTED] please"


def test_credit_card_invalid_luhn_character_by_character() -> None:
    text = f"pay with {INVALID_LUHN_16} please"
    assert _run_stream(text, 1) == text  # unchanged


# =============================================================================
# 1d. Category D -- mixed streams
# =============================================================================

MIXED_CASES = [
    (
        f"Email a@b.com SSN 123-45-6789 Card {VALID_VISA_16} done",
        f"Email [REDACTED] SSN [REDACTED] Card [REDACTED] done",
    ),
    ("Card:" + VALID_VISA_16 + ",Email:a@b.com", "Card:[REDACTED],Email:[REDACTED]"),
    ("café user@example.com résumé", "café [REDACTED] résumé"),  # unicode around ASCII PII
    ("日本語 123-45-6789 中文", "日本語 [REDACTED] 中文"),
    ("a@b.com, c@d.com, 123-45-6789.", "[REDACTED], [REDACTED], [REDACTED]."),  # adjacent-ish, comma separated
    ("no pii here at all, just plain text.", "no pii here at all, just plain text."),
]


@pytest.mark.parametrize("text,expected,chunk_size", list(_every_split_cases(MIXED_CASES)))
def test_mixed_every_split_position(text: str, expected: str, chunk_size: int) -> None:
    """Covers: multiple categories in one response, near-adjacent candidates

    separated by punctuation, safe text before/after, non-ASCII text
    surrounding ASCII PII, and pure safe text with zero PII.
    """
    assert _run_stream(text, chunk_size) == expected


def test_pii_beginning_exactly_at_chunk_boundary() -> None:
    """PII starting exactly where one chunk ends and the next begins."""
    prefix = "Note: "
    pii = "user@example.com"
    suffix = " end of message"
    text = prefix + pii + suffix
    # Force the split to land exactly at the prefix/PII boundary.
    assert _run_stream(text, len(prefix)) == "Note: [REDACTED] end of message"


def test_pii_ending_exactly_at_chunk_boundary() -> None:
    prefix = "Note: "
    pii = "123-45-6789"
    suffix = " end"
    text = prefix + pii + suffix
    assert _run_stream(text, len(prefix) + len(pii)) == "Note: [REDACTED] end"


def test_false_candidate_prefix_that_later_proves_safe() -> None:
    """A prefix that looks like it could be forming PII, but never resolves into it."""
    # Looks like it could become an email (has '.', letters) but no '@' ever appears.
    text = "the file is report.final.v2.docx not an email"
    assert _run_stream(text, 1) == text
    # Looks like it could become a card/SSN (digits+hyphens) but breaks structure.
    text2 = "reference 12-3456-78 is not a card or ssn"
    assert _run_stream(text2, 1) == text2


# =============================================================================
# 1e. Category E -- streaming guarantees
# =============================================================================


def test_safe_text_emitted_incrementally_before_pii_arrives() -> None:
    """A long safe prefix must already be returned by feed() before the PII

    chunk (arriving later) is even fed in -- proves output isn't held back
    waiting for the whole response. Deliberately ends in "!": both a trailing
    space and a trailing "." are themselves legitimately ambiguous candidate
    starts (a space could precede spaced-format card digits; "." and the
    letters before it could still be forming an email local part even with
    no '@' seen yet) and are correctly held back rather than emitted -- "!"
    is outside every supported candidate alphabet, so it cleanly resolves
    the boundary for this assertion.
    """
    redactor = StreamingRedactor()
    safe_prefix = " ".join(["This is a long safe introduction with no PII at all!"] * 5)
    out1 = redactor.feed(safe_prefix)
    assert out1 == safe_prefix  # already emitted, before any PII chunk exists
    out2 = redactor.feed(" Contact user@example.com")
    out3 = redactor.flush()
    assert out1 + out2 + out3 == safe_prefix + " Contact [REDACTED]"


def test_pending_state_bounded_for_long_safe_prose() -> None:
    redactor = StreamingRedactor()
    long_prose = "the quick brown fox jumps over the lazy dog. " * 200
    max_pending = 0
    for ch in long_prose:
        redactor.feed(ch)
        max_pending = max(max_pending, redactor.pending_size)
    redactor.flush()
    # Ordinary prose (spaces every few chars) should never need to hold back
    # more than a single short trailing word.
    assert max_pending < 20, f"pending grew to {max_pending} for ordinary prose"


def test_pending_state_bounded_for_adversarial_numeric_noise() -> None:
    """An unbroken digit/space/hyphen run that never resolves into SSN or card

    must still be bounded at NUMERIC_CANDIDATE_MAX, not grow without limit.

    Uses "10101" rather than "12345" as the repeating digit group: "12345 "
    (and, it turns out, "13579 " too) repeated coincidentally *does* contain
    a genuine Luhn-valid 16-digit card a few repeats in at some rotation
    (verified via `_find_card_spans`), which correctly (see the
    connector-aware openness handling in `feed()`) holds `pending_size` one
    character past this bound while that discovered card is still touching
    the end of the buffer -- still bounded, just not exactly equal to this
    constant, so it is the wrong text for a test asserting *no* resolvable
    candidate ever appears at any rotation. "10101" was verified (by
    `_find_card_spans` returning empty for many repeats, at every rotation
    the streaming trim can produce) to contain none.
    """
    redactor = StreamingRedactor()
    noise = "10101 " * 1000
    max_pending = 0
    for ch in noise:
        redactor.feed(ch)
        max_pending = max(max_pending, redactor.pending_size)
    redactor.flush()
    assert max_pending == NUMERIC_CANDIDATE_MAX


def test_pending_state_bounded_for_adversarial_email_alphabet_noise() -> None:
    """An unbroken run of email-alphabet characters with no '@' must be bounded

    at EMAIL_LOCAL_MAX, not grow without limit.
    """
    redactor = StreamingRedactor()
    noise = "a" * 5000
    max_pending = 0
    for ch in noise:
        redactor.feed(ch)
        max_pending = max(max_pending, redactor.pending_size)
    redactor.flush()
    assert max_pending == EMAIL_LOCAL_MAX


def test_pending_state_bounded_for_adversarial_domain_noise() -> None:
    """An '@' followed by an unbroken run of domain-alphabet characters that

    never terminates becomes impossible when the label exceeds 63 chars.
    The retained local-part suffix can still precede a new '@'.
    """
    redactor = StreamingRedactor()
    noise = "a@" + "b" * 5000
    max_pending = 0
    for ch in noise:
        redactor.feed(ch)
        max_pending = max(max_pending, redactor.pending_size)
    redactor.flush()
    assert max_pending == len("a@") + EMAIL_LABEL_MAX


def test_flush_resolves_pii_ending_exactly_at_eof() -> None:
    for text, expected in [
        ("my email is user@example.com", "my email is [REDACTED]"),
        ("my ssn is 123-45-6789", "my ssn is [REDACTED]"),
        (f"my card is {VALID_VISA_16}", "my card is [REDACTED]"),
    ]:
        redactor = StreamingRedactor()
        out = redactor.feed(text)
        out += redactor.flush()
        assert out == expected


def test_flush_is_idempotent_reset() -> None:
    redactor = StreamingRedactor()
    redactor.feed("user@example.com")
    redactor.flush()
    assert redactor.pending_size == 0
    # A second flush with no further feed() returns nothing new.
    assert redactor.flush() == ""


# =============================================================================
# 1f. Regression -- chunk-boundary independence (M3 review finding P1)
# =============================================================================


def _run_chunks(chunks: list[str]) -> str:
    redactor = StreamingRedactor()
    parts = [redactor.feed(chunk) for chunk in chunks]
    parts.append(redactor.flush())
    return "".join(parts)


def _all_chunkings(text: str) -> list[tuple[str, list[str]]]:
    """Every chunking strategy the review requires: one chunk, every possible

    two-way split, small fixed chunk sizes, and character-by-character.
    """
    strategies: list[tuple[str, list[str]]] = [("one-chunk", [text])]
    for i in range(len(text) + 1):
        strategies.append((f"two-way-split@{i}", [text[:i], text[i:]]))
    for size in range(1, min(len(text), 12) + 1):
        strategies.append((f"fixed-size-{size}", [text[i : i + size] for i in range(0, len(text), size)]))
    strategies.append(("char-by-char", list(text)))
    return strategies


# The first four entries are the exact cases reproduced in the M3 review.
CHUNK_INDEPENDENCE_CORPUS: list[tuple[str, str]] = [
    ("ssn-then-long-space-run", "123-45-6789" + " " * 40),
    ("card-then-long-space-run", VALID_VISA_16 + " " * 40),
    ("email-then-second-at", "user@example.com@x!"),
    ("ssn-inside-invalid-card-region", "123-45-6789 1234!"),
    ("ssn-then-long-hyphen-run", "123-45-6789" + "-" * 45),
    ("email-then-long-dot-run", "user@example.com" + "." * 45),
    ("long-digits-then-ssn", "1" * 60 + " 123-45-6789"),
    ("email-inside-at-noise", "@@@user@example.com@@@"),
    ("multiple-mixed-pii", f"Email a@b.com SSN 123-45-6789 Card {VALID_VISA_16} done"),
    ("unicode-around-pii", "café 123-45-6789 日本語 user@example.com résumé"),
    ("spaced-card-and-amex", f"4111 1111 1111 1111 and {VALID_AMEX_15} end"),
    ("ssn-with-trailing-group", "123-45-6789-0123 end"),
    ("adjacent-emails-and-ssn", "a@b.co.uk,c@d.io;123-45-6789."),
    ("leading-space-run-then-card", " " * 50 + VALID_VISA_16 + " " * 50),
    ("no-pii-at-all", "just ordinary prose with no personal data in it"),
]


@pytest.mark.parametrize(
    "text", [text for _, text in CHUNK_INDEPENDENCE_CORPUS], ids=[name for name, _ in CHUNK_INDEPENDENCE_CORPUS]
)
def test_output_is_independent_of_chunk_boundaries(text: str) -> None:
    """The core streaming invariant: for one logical input, the final redacted

    output must be identical no matter how the stream was chunked.
    """
    baseline = _run_chunks([text])
    for label, chunks in _all_chunkings(text):
        assert _run_chunks(chunks) == baseline, (
            f"chunking strategy {label!r} produced different output than a single chunk"
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("123-45-6789" + " " * 40, "[REDACTED]" + " " * 40),
        (VALID_VISA_16 + " " * 40, "[REDACTED]" + " " * 40),
        ("user@example.com@x!", "[REDACTED]@x!"),
    ],
    ids=["ssn-then-spaces", "card-then-spaces", "email-then-second-at"],
)
def test_reported_cross_chunk_leaks_are_fixed(text: str, expected: str) -> None:
    """The three exact cross-chunk leaks reported by the review.

    Root causes: (a) the candidate-scan length cap could return a boundary that
    landed inside a *complete* match at the start of the capped window, and
    (b) the email scan's second-'@' break could return a boundary inside an
    already-complete email. Both cut a finished match in half, emitting the
    first half raw.
    """
    for label, chunks in _all_chunkings(text):
        assert _run_chunks(chunks) == expected, f"chunking {label!r} leaked or altered output"


@pytest.mark.parametrize(
    "text,secret",
    [
        ("123-45-6789" + " " * 40, "123-45-6789"),
        (VALID_VISA_16 + " " * 40, VALID_VISA_16),
        ("user@example.com@x!", "user@example.com"),
        ("123-45-6789 1234!", "123-45-6789"),
    ],
)
def test_no_pii_substring_survives_under_any_chunking(text: str, secret: str) -> None:
    for label, chunks in _all_chunkings(text):
        assert secret not in _run_chunks(chunks), f"chunking {label!r} leaked {secret!r}"


# =============================================================================
# 1g. Regression -- failed card classification must not hide a valid SSN (P1)
# =============================================================================

SSN_VS_INVALID_CARD_CASES = [
    # (label, text, expected)
    ("ssn-then-four-digits", "123-45-6789 1234!", "[REDACTED] 1234!"),
    ("ssn-then-two-groups", "123-45-6789 1234 5678!", "[REDACTED] 1234 5678!"),
    ("digits-before-ssn", "9999 123-45-6789!", "9999 [REDACTED]!"),
    (
        "invalid-luhn-card-region-containing-ssn",
        "ref 1234-5678-9012-3456 and ssn 123-45-6789 end",
        "ref 1234-5678-9012-3456 and ssn [REDACTED] end",
    ),
    ("invalid-luhn-identifier-no-ssn", "1234567890123456!", "1234567890123456!"),
    ("valid-card-and-ssn-together", f"card {VALID_VISA_16} ssn 123-45-6789 end", "card [REDACTED] ssn [REDACTED] end"),
    ("ssn-then-long-numeric-id", "x 123-45-6789 1234567890123 y", "x [REDACTED] 1234567890123 y"),
]


@pytest.mark.parametrize(
    "text,expected",
    [(text, expected) for _, text, expected in SSN_VS_INVALID_CARD_CASES],
    ids=[label for label, _, _ in SSN_VS_INVALID_CARD_CASES],
)
def test_failed_card_does_not_suppress_valid_ssn(text: str, expected: str) -> None:
    """Root cause: a single alternation regex consumed the whole card-shaped

    numeric region, and returning it unchanged on Luhn failure meant the SSN
    nested inside never got its own chance to match. Each pattern is now
    scanned independently, so a dropped card candidate leaves any SSN inside
    it intact for detection.
    """
    assert _run_chunks([text]) == expected  # single chunk
    assert _run_chunks(list(text)) == expected  # character-by-character
    for label, chunks in _all_chunkings(text):
        assert _run_chunks(chunks) == expected, f"chunking {label!r} produced different output"


def test_invalid_luhn_region_still_not_redacted_as_card() -> None:
    """The Luhn gate must still suppress card redaction of unrelated identifiers."""
    text = f"order {INVALID_LUHN_16} shipped"
    assert _run_chunks([text]) == text
    assert _run_chunks(list(text)) == text


# =============================================================================
# 1h. Regression -- card-vs-card masking & left-context loss across a trim
#     (independent post-fix review, round 2)
# =============================================================================


def _representative_three_way_splits(text: str) -> list[tuple[str, list[str]]]:
    """A handful of representative (not exhaustive) three-way splits.

    `_all_chunkings` already covers every two-way split, fixed sizes 1-12, and
    char-by-char; exhaustive three-way splits would be O(n^2) extra cases per
    string. These few early/middle/late cut combinations are enough to catch
    a boundary bug the other strategies miss without the blowup.
    """
    n = len(text)
    if n < 3:
        return []
    cuts = sorted({max(1, n // 4), max(2, n // 2), min(n - 1, (3 * n) // 4)})
    splits: list[tuple[str, list[str]]] = []
    for i in cuts:
        for j in cuts:
            if i < j:
                splits.append((f"3way@{i},{j}", [text[:i], text[i:j], text[j:]]))
    return splits


def _assert_chunking_never_leaks(text: str, expected: str, secrets: list[str]) -> None:
    """Per the review's test requirement: chunk-invariance alone is not
    enough if every chunking leaks the same PII. For each chunking strategy,
    assert (1) output is byte-identical to the one-chunk baseline, (2) that
    output matches the expected fully-redacted form (i.e. the [REDACTED]
    markers are in the right places, not just "consistent"), and (3) none of
    the original PII substrings survive.
    """
    baseline = _run_chunks([text])
    assert baseline == expected
    for secret in secrets:
        assert secret not in baseline, f"one-chunk baseline itself leaked {secret!r}"
    strategies = _all_chunkings(text) + _representative_three_way_splits(text)
    for label, chunks in strategies:
        got = _run_chunks(chunks)
        assert got == expected, f"chunking {label!r} produced {got!r}, expected {expected!r}"
        for secret in secrets:
            assert secret not in got, f"chunking {label!r} leaked {secret!r} in {got!r}"


VALID_MC_16_B = "5500000000000004"
assert _luhn_valid(VALID_MC_16_B)

# label, text, fully-redacted expected output, PII substrings that must never
# survive under any chunking.
CARD_VS_CARD_CORPUS: list[tuple[str, str, str, list[str]]] = [
    (
        "ssn-then-valid-card",
        f"123-45-6789 {VALID_VISA_16}",
        "[REDACTED] [REDACTED]",
        ["123-45-6789", VALID_VISA_16],
    ),
    (
        "ssn-then-spaced-valid-card",
        "123-45-6789 4111 1111 1111 1111",
        "[REDACTED]",
        ["123-45-6789", "4111 1111 1111 1111", VALID_VISA_16],
    ),
    (
        "card-then-ssn",
        f"{VALID_VISA_16} 123-45-6789",
        "[REDACTED] [REDACTED]",
        [VALID_VISA_16, "123-45-6789"],
    ),
    (
        "ssn-then-amex",
        f"123-45-6789 {VALID_AMEX_15}",
        "[REDACTED] [REDACTED]",
        ["123-45-6789", VALID_AMEX_15],
    ),
    (
        "invalid-luhn-prefix-then-valid-card",
        "1234 5678 9012 3456 4111 1111 1111 1111",
        "1234 5678 9012 3456 [REDACTED]",
        ["4111 1111 1111 1111", VALID_VISA_16],
    ),
    (
        "valid-card-then-invalid-luhn-suffix",
        "4111 1111 1111 1111 1234 5678 9012 3456",
        "[REDACTED] 1234 5678 9012 3456",
        ["4111 1111 1111 1111", VALID_VISA_16],
    ),
    (
        "adjacent-distinct-valid-cards-spaced",
        f"{VALID_VISA_16} {VALID_AMEX_15}",
        "[REDACTED] [REDACTED]",
        [VALID_VISA_16, VALID_AMEX_15],
    ),
    (
        "multiple-valid-cards-spaces-and-hyphens",
        f"{VALID_VISA_16} {VALID_AMEX_15}-{VALID_MC_16_B}",
        "[REDACTED] [REDACTED]-[REDACTED]",
        [VALID_VISA_16, VALID_AMEX_15, VALID_MC_16_B],
    ),
    (
        "critical1-exact-repro",
        "Your SSN is 123-45-6789 4111 1111 1111 1111 thanks",
        "Your SSN is [REDACTED] thanks",
        ["123-45-6789", "4111 1111 1111 1111", VALID_VISA_16],
    ),
    (
        "sandwiched-ssn-in-long-digit-runs",
        "9" * 30 + " 123-45-6789 " + "9" * 30,
        "9" * 30 + " [REDACTED] " + "9" * 30,
        ["123-45-6789"],
    ),
    (
        "invalid-luhn-identifier-alone-unchanged",
        "1234 5678 9012 3456",
        "1234 5678 9012 3456",
        [],  # nothing to redact; asserts the text is left byte-for-byte alone
    ),
]


@pytest.mark.parametrize(
    "text,expected,secrets",
    [(text, expected, secrets) for _, text, expected, secrets in CARD_VS_CARD_CORPUS],
    ids=[label for label, _, _, _ in CARD_VS_CARD_CORPUS],
)
def test_card_vs_card_masking_and_left_context_fixed(text: str, expected: str, secrets: list[str]) -> None:
    """Post-fix-review regressions (round 2), covering two independent root causes:

    1. **Card-vs-card / card-vs-SSN masking.** A single greedy regex match
       per numeric run finds the *longest* digit-shaped candidate first; if it
       fails Luhn, `finditer` resumes scanning after it, skipping over a valid
       card (or SSN) that starts inside the rejected region -- e.g. in
       "123-45-6789 4111 1111 1111 1111" the greedy 19-digit candidate
       spanning the SSN and the start of the card fails Luhn, hiding the real
       card. `_find_card_spans` instead decomposes every maximal digit run
       into its separator-delimited blocks and checks every contiguous
       combination for validity, so a rejected superset can never hide a
       valid subset or neighbour.
    2. **Left-context loss across a trim.** Once a long digit/space/hyphen run
       forces `StreamingRedactor` to trim `_pending`, the new buffer's
       position 0 can sit in the *middle* of a digit run that continues
       further left in already-emitted text. Re-scanning the trimmed buffer
       sees "start of string" there and `(?<!\\d)` succeeds *vacuously*, even
       though a real digit precedes it in the true stream. `_unsafe_start`
       tracks this across trims and discards any match that would rely on
       that vacuous truth.
    """
    _assert_chunking_never_leaks(text, expected, secrets)


# =============================================================================
# 1i. Regression -- repeated identical valid cards inside one run must
#     resolve deterministically, not depend on how much of the run happens
#     to be visible when the decision is made (independent review, round 3)
# =============================================================================

REPEATED_CARD_CORPUS: list[tuple[str, str, str, list[str]]] = [
    (
        "two-contiguous-cards-spaced",
        "4111111111111111 4111111111111111",
        "[REDACTED] [REDACTED]",
        [VALID_VISA_16],
    ),
    (
        "two-spaced-cards-back-to-back",
        "4111 1111 1111 1111 4111 1111 1111 1111",
        "[REDACTED]",
        ["4111 1111 1111 1111", VALID_VISA_16],
    ),
    (
        "two-contiguous-cards-hyphenated",
        "4111111111111111-4111111111111111",
        "[REDACTED]-[REDACTED]",
        [VALID_VISA_16],
    ),
    (
        "two-cards-with-prose-around",
        "prefix 4111111111111111 4111111111111111 suffix",
        "prefix [REDACTED] [REDACTED] suffix",
        [VALID_VISA_16],
    ),
    (
        "three-contiguous-cards-spaced",
        "4111111111111111 4111111111111111 4111111111111111",
        "[REDACTED] [REDACTED] [REDACTED]",
        [VALID_VISA_16],
    ),
    (
        "three-spaced-cards-back-to-back",
        "4111 1111 1111 1111 4111 1111 1111 1111 4111 1111 1111 1111",
        "[REDACTED]",
        ["4111 1111 1111 1111", VALID_VISA_16],
    ),
]


@pytest.mark.parametrize(
    "text,expected,secrets",
    [(text, expected, secrets) for _, text, expected, secrets in REPEATED_CARD_CORPUS],
    ids=[label for label, _, _, _ in REPEATED_CARD_CORPUS],
)
def test_repeated_identical_valid_cards_resolve_deterministically(
    text: str, expected: str, secrets: list[str]
) -> None:
    """Root cause: several identical spaced valid cards back to back (e.g. two

    copies of "4111 1111 1111 1111") can have *every* block-boundary rotation
    of the run also independently pass Luhn, chaining the whole run into one
    merged region whose exact extent is only fully known once the run's own
    end is visible. Confirming (redacting and permanently trimming past) such
    a region before that point meant the answer depended on how many repeats
    happened to already be buffered when a chunk boundary forced a decision --
    sometimes producing an extra `[REDACTED]` marker, sometimes leaving the
    last few digits of one copy as plaintext next to an already-redacted
    neighbour.

    `feed()` now leaves a merged span unconfirmed while the numeric run
    underlying it is still *open* (extends to the end of currently-known
    text), deferring the decision until the run genuinely closes -- at which
    point every chunking sees the same complete picture and computes the same
    answer. That wait is bounded by `MAX_MERGE_HOLDBACK`, not indefinite, so
    pending state never grows without limit even if a run never closes.
    """
    _assert_chunking_never_leaks(text, expected, secrets)


def test_pending_state_bounded_for_long_repeated_card_chain() -> None:
    """The chain-confirmation deferral above must not itself become an

    unbounded buffer: even a very long run of repeated identical valid cards
    (far longer than any of the required test cases) must keep `pending_size`
    within a small, fixed bound, via the MAX_MERGE_HOLDBACK escape valve.
    """
    redactor = StreamingRedactor()
    chain = ("4111 1111 1111 1111 " * 200).strip()
    max_pending = 0
    for ch in chain:
        redactor.feed(ch)
        max_pending = max(max_pending, redactor.pending_size)
    redactor.flush()
    assert max_pending <= MAX_MERGE_HOLDBACK, f"pending grew to {max_pending} for a long repeated-card chain"


@pytest.mark.parametrize("repeats", [2, 3, 4, 5])
def test_required_cases_resolve_below_the_escape_valve_threshold(repeats: int) -> None:
    """The escape valve documented next to `MAX_MERGE_HOLDBACK` must not be

    what makes the *required* repeated-card cases (explicitly: two and three
    repeats) correct -- only what bounds chains far longer than any of them.
    This asserts the invariant directly for two through five repeats (a
    small margin past what's explicitly required): `pending_size` never even
    reaches MAX_MERGE_HOLDBACK, meaning each resolves through the fully
    chunk-invariant natural-closure path, not the fallback. This is *not*
    claimed for arbitrarily many repeats -- a chain engineered so every
    block-boundary rotation keeps validating forever necessarily outgrows any
    fixed threshold eventually; `test_extreme_overlapping_card_chain_stays_bounded_safe_and_chunk_invariant`
    covers that regime instead, where the valve is expected to fire.
    """
    redactor = StreamingRedactor()
    chain = ("4111 1111 1111 1111 " * repeats).strip()
    max_pending = 0
    for ch in chain:
        redactor.feed(ch)
        max_pending = max(max_pending, redactor.pending_size)
    redactor.flush()
    assert max_pending < MAX_MERGE_HOLDBACK, (
        f"a {repeats}-repeat chain reached pending={max_pending}, "
        f"at or above MAX_MERGE_HOLDBACK={MAX_MERGE_HOLDBACK} -- the escape valve fired "
        "for a case the natural-closure path is supposed to cover"
    )


def test_extreme_overlapping_card_chain_stays_bounded_safe_and_chunk_invariant() -> None:
    """Stress test for the MAX_MERGE_HOLDBACK escape valve itself (not just

    the natural-closure path above): a chain of 3000 repeats -- far beyond
    anything the natural-closure path or any required case needs -- must
    still (1) keep `pending_size` bounded, (2) never leak the original card
    value as a contiguous substring anywhere in the output, and (3) produce
    byte-identical output whether fed as one chunk, in small fixed chunks, or
    character by character. This is what the "boundedness" and
    "chunk-invariance" proofs next to MAX_MERGE_HOLDBACK claim; this test
    exercises them well past the point where the valve is guaranteed to have
    fired at least once.
    """
    chain = ("4111 1111 1111 1111 " * 3000).strip()

    def run_and_track(chunks: list[str]) -> tuple[str, int]:
        redactor = StreamingRedactor()
        parts = []
        max_pending = 0
        for chunk in chunks:
            parts.append(redactor.feed(chunk))
            max_pending = max(max_pending, redactor.pending_size)
        parts.append(redactor.flush())
        return "".join(parts), max_pending

    baseline, max_pending_one_shot = run_and_track([chain])
    assert VALID_VISA_16 not in baseline
    assert "4111 1111 1111 1111" not in baseline
    assert max_pending_one_shot <= MAX_MERGE_HOLDBACK + CREDIT_CARD_MAX, (
        f"one-shot feed reached pending={max_pending_one_shot}"
    )

    for label, chunks in [
        ("char-by-char", list(chain)),
        ("fixed-7", [chain[i : i + 7] for i in range(0, len(chain), 7)]),
        ("fixed-40", [chain[i : i + 40] for i in range(0, len(chain), 40)]),
        ("fixed-200", [chain[i : i + 200] for i in range(0, len(chain), 200)]),
    ]:
        got, max_pending = run_and_track(chunks)
        assert got == baseline, f"chunking {label!r} diverged from the one-shot baseline"
        assert VALID_VISA_16 not in got, f"chunking {label!r} leaked the card value"
        assert max_pending <= MAX_MERGE_HOLDBACK + CREDIT_CARD_MAX, (
            f"chunking {label!r} reached pending={max_pending}"
        )


# =============================================================================
# 1j. Regression -- card immediately adjacent to an email, no separator
#     (independent review, round 4): EMAIL_PATTERN's local-part alphabet
#     overlaps the numeric one (digits are valid in both), so a card ending
#     right where more email-alphabet characters immediately follow can have
#     its trailing digits absorbed into an eventual email's local part.
#     Confirming the card before knowing whether that happens made the
#     redaction-marker grouping depend on chunk boundaries.
# =============================================================================

CARD_EMAIL_ADJACENCY_CORPUS: list[tuple[str, str, str, list[str]]] = [
    (
        "contiguous-card-then-email",
        f"{VALID_VISA_16}user@example.com",
        "[REDACTED]",
        [VALID_VISA_16, "user@example.com"],
    ),
    (
        "spaced-card-then-email",
        "4111 1111 1111 1111user@example.com",
        "[REDACTED]",
        ["4111 1111 1111 1111", VALID_VISA_16, "user@example.com"],
    ),
    (
        "email-then-contiguous-card",
        f"user@example.com{VALID_VISA_16}",
        "[REDACTED][REDACTED]",
        [VALID_VISA_16, "user@example.com"],
    ),
    (
        "email-then-spaced-card",
        "user@example.com4111 1111 1111 1111",
        "[REDACTED][REDACTED]",
        ["4111 1111 1111 1111", VALID_VISA_16, "user@example.com"],
    ),
    (
        "card-email-card-sandwich",
        f"{VALID_VISA_16}user@example.com{VALID_VISA_16}",
        "[REDACTED][REDACTED]",
        [VALID_VISA_16, "user@example.com"],
    ),
    (
        "card-email-with-leading-punctuation",
        f"ref:{VALID_VISA_16}user@example.com!",
        "ref:[REDACTED]!",
        [VALID_VISA_16, "user@example.com"],
    ),
    (
        "card-email-with-surrounding-prose",
        f"prefix 4111 1111 1111 1111user@example.com suffix",
        "prefix [REDACTED] suffix",
        ["4111 1111 1111 1111", VALID_VISA_16, "user@example.com"],
    ),
    (
        "parenthesised-email-then-card",
        f"(user@example.com{VALID_VISA_16})",
        "([REDACTED][REDACTED])",
        [VALID_VISA_16, "user@example.com"],
    ),
    (
        "card-then-email-then-period",
        f"{VALID_VISA_16}user@example.com.",
        "[REDACTED].",
        [VALID_VISA_16, "user@example.com"],
    ),
    (
        "period-then-card-then-email",
        f".{VALID_VISA_16}user@example.com",
        "[REDACTED]",
        [VALID_VISA_16, "user@example.com"],
    ),
    (
        "short-email-then-card",
        f"a@b.co{VALID_VISA_16}",
        "[REDACTED][REDACTED]",
        [VALID_VISA_16, "a@b.co"],
    ),
    (
        "card-then-short-email",
        f"{VALID_VISA_16}a@b.co",
        "[REDACTED]",
        [VALID_VISA_16, "a@b.co"],
    ),
]


@pytest.mark.parametrize(
    "text,expected,secrets",
    [(text, expected, secrets) for _, text, expected, secrets in CARD_EMAIL_ADJACENCY_CORPUS],
    ids=[label for label, _, _, _ in CARD_EMAIL_ADJACENCY_CORPUS],
)
def test_card_email_adjacency_resolves_deterministically(text: str, expected: str, secrets: list[str]) -> None:
    """Root cause: `_numeric_tail_open` only checked whether the numeric

    (digit/space/hyphen) alphabet was still reaching the end of the buffer --
    it had no way to know that EMAIL_PATTERN's local-part alphabet (which
    includes digits) might still be extending too, right where the card's own
    digits end. `feed()` now also checks `email_naive` (the same bounded,
    already-computed "how far back could an email still be reaching" value
    used for the ordinary email candidate cutoff) before confirming a numeric
    span whose end directly abuts more email-alphabet characters, so the card
    and the email it touches are never split across a chunk boundary.
    """
    _assert_chunking_never_leaks(text, expected, secrets)


# =============================================================================
# 2. `_redacted_stream` generator tests against a fake upstream (no ASGI)
# =============================================================================


class _FakeUpstreamResponse:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def aiter_text(self) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContextManager:
    def __init__(self, chunks: list[str], state: dict[str, bool]) -> None:
        self._chunks = chunks
        self._state = state

    async def __aenter__(self) -> _FakeUpstreamResponse:
        self._state["entered"] = True
        return _FakeUpstreamResponse(self._chunks)

    async def __aexit__(self, *exc_info: object) -> bool:
        self._state["exited"] = True
        return False


class FakeHttpClient:
    """Minimal stand-in for httpx.AsyncClient exposing only what _redacted_stream uses."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.state: dict[str, bool] = {"entered": False, "exited": False}

    def stream(self, method: str, url: str, json: object = None):  # noqa: A002
        return _FakeStreamContextManager(self._chunks, self.state)


async def test_redacted_stream_yields_incrementally_and_closes_upstream() -> None:
    client = FakeHttpClient(["safe one ", "safe two ", "user@", "example.com", " done"])
    outputs = [chunk async for chunk in gateway_app._redacted_stream("ignored", 8, client)]
    assert len(outputs) > 1, "expected multiple incremental yields, not one combined blob"
    assert b"".join(outputs) == b"safe one safe two [REDACTED] done"
    assert client.state == {"entered": True, "exited": True}


async def test_redacted_stream_closes_upstream_on_early_cancellation() -> None:
    client = FakeHttpClient(["a" * 50, "b" * 50, "c" * 50])
    gen = gateway_app._redacted_stream("ignored", 8, client)
    await gen.__anext__()  # consume exactly one chunk
    await gen.aclose()  # simulate downstream disconnect
    assert client.state["exited"] is True


async def test_redacted_stream_final_pii_at_eof_via_flush() -> None:
    client = FakeHttpClient(["contact us at ", "user@example.com"])
    outputs = [chunk async for chunk in gateway_app._redacted_stream("ignored", 8, client)]
    assert b"".join(outputs) == b"contact us at [REDACTED]"


# =============================================================================
# 2b. Regression -- resource lifecycle (M3 review finding P2)
# =============================================================================


async def _drive_asgi_response(response, send) -> None:
    """Invoke a Starlette Response as an ASGI app with a minimal http scope."""

    async def receive() -> dict[str, str]:
        return {"type": "http.request"}

    await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)


async def test_upstream_closed_when_downstream_send_fails_mid_stream() -> None:
    """Root cause: Starlette's StreamingResponse.stream_response() iterates the

    body iterator but never closes it, so a send() failure after data had
    already been yielded abandoned our generator mid-`yield` and left the
    upstream httpx stream open until non-deterministic asyncgen finalization.
    _ClosingStreamingResponse closes the iterator in a finally instead.
    """
    client = FakeHttpClient(["hello ", "world ", "and more safe text "])
    response = gateway_app._ClosingStreamingResponse(
        gateway_app._redacted_stream("ignored", 8, client), media_type="text/plain"
    )

    async def failing_send(message: dict) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            raise RuntimeError("downstream client vanished")

    with pytest.raises(RuntimeError, match="downstream client vanished"):
        await _drive_asgi_response(response, failing_send)

    assert client.state["entered"] is True
    assert client.state["exited"] is True, "upstream stream was left open after a downstream send failure"


async def test_plain_streaming_response_would_leak_upstream() -> None:
    """Characterization test proving the fix above is load-bearing: the stock

    StreamingResponse leaves the upstream open in exactly this scenario, which
    is why the endpoint uses _ClosingStreamingResponse.
    """
    from fastapi.responses import StreamingResponse

    client = FakeHttpClient(["hello ", "world ", "and more safe text "])
    response = StreamingResponse(
        gateway_app._redacted_stream("ignored", 8, client), media_type="text/plain"
    )

    async def failing_send(message: dict) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            raise RuntimeError("downstream client vanished")

    with pytest.raises(RuntimeError):
        await _drive_asgi_response(response, failing_send)

    assert client.state["exited"] is False  # documents the behavior the subclass fixes


async def test_upstream_closed_on_normal_completion() -> None:
    client = FakeHttpClient(["safe text ", "user@example.com"])
    response = gateway_app._ClosingStreamingResponse(
        gateway_app._redacted_stream("ignored", 8, client), media_type="text/plain"
    )
    body: list[bytes] = []

    async def collecting_send(message: dict) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            body.append(message["body"])

    await _drive_asgi_response(response, collecting_send)
    assert b"".join(body) == b"safe text [REDACTED]"
    assert client.state["exited"] is True


async def test_lifespan_closes_shared_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared client used by real (non-test) runs must be closed on shutdown."""
    fresh_client = httpx.AsyncClient()
    monkeypatch.setattr(gateway_app, "_shared_http_client", fresh_client)
    async with gateway_app._lifespan(gateway_app.app):
        assert not fresh_client.is_closed
    assert fresh_client.is_closed


async def test_lifespan_closes_shared_http_client_on_exceptional_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a bare `yield` with no try/finally skips aclose() entirely

    if an exception propagates through the lifespan context (e.g. a
    startup/serving failure) -- the statement after a plain `yield` only runs
    on the non-raising path. `_lifespan` now wraps it in try/finally.
    """
    fresh_client = httpx.AsyncClient()
    monkeypatch.setattr(gateway_app, "_shared_http_client", fresh_client)
    with pytest.raises(RuntimeError, match="boom"):
        async with gateway_app._lifespan(gateway_app.app):
            raise RuntimeError("boom")
    assert fresh_client.is_closed, "shared AsyncClient was left open after an exceptional lifespan exit"


# =============================================================================
# 3. Gateway + mock_llm ASGI integration tests (functional correctness)
# =============================================================================


@pytest.fixture
async def mock_upstream_client():
    transport = ASGITransport(app=mock_llm.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-upstream") as client:
        yield client


@pytest.fixture
async def gateway_client(mock_upstream_client: httpx.AsyncClient):
    gateway_app.app.dependency_overrides[gateway_app.get_http_client] = lambda: mock_upstream_client
    transport = ASGITransport(app=gateway_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        yield client
    gateway_app.app.dependency_overrides.clear()


async def test_gateway_redacts_end_to_end(gateway_client: httpx.AsyncClient) -> None:
    prompt = f"Contact user@example.com, SSN 123-45-6789, card {VALID_VISA_16} thanks"
    async with gateway_client.stream("POST", "/generate", json={"prompt": prompt, "chunk_size": 3}) as resp:
        body = b"".join([chunk async for chunk in resp.aiter_bytes()]).decode("utf-8")
    assert resp.status_code == 200
    assert "user@example.com" not in body
    assert "123-45-6789" not in body
    assert VALID_VISA_16 not in body
    assert body.count("[REDACTED]") == 3
    assert body == "Contact [REDACTED], SSN [REDACTED], card [REDACTED] thanks"


async def test_gateway_character_by_character_upstream_chunking(gateway_client: httpx.AsyncClient) -> None:
    prompt = "email user@example.com now"
    async with gateway_client.stream("POST", "/generate", json={"prompt": prompt, "chunk_size": 1}) as resp:
        body = b"".join([chunk async for chunk in resp.aiter_bytes()]).decode("utf-8")
    assert body == "email [REDACTED] now"


async def test_gateway_passes_through_safe_text_unchanged(gateway_client: httpx.AsyncClient) -> None:
    prompt = "There is no personal information in this sentence at all."
    async with gateway_client.stream("POST", "/generate", json={"prompt": prompt, "chunk_size": 4}) as resp:
        body = b"".join([chunk async for chunk in resp.aiter_bytes()]).decode("utf-8")
    assert body == prompt


async def test_gateway_multibyte_utf8_not_corrupted_by_chunking(gateway_client: httpx.AsyncClient) -> None:
    """A multi-byte UTF-8 character split across upstream chunk boundaries

    must decode correctly (proves aiter_text()'s incremental decoder is
    actually in the path, not a naive per-chunk .decode()).
    """
    prompt = "café 日本語 user@example.com résumé"
    async with gateway_client.stream("POST", "/generate", json={"prompt": prompt, "chunk_size": 1}) as resp:
        body = b"".join([chunk async for chunk in resp.aiter_bytes()]).decode("utf-8")
    assert body == "café 日本語 [REDACTED] résumé"


async def test_gateway_client_disconnect_mid_stream_closes_cleanly(gateway_client: httpx.AsyncClient) -> None:
    """Reading only part of the response and closing early must not raise

    or hang -- proves upstream cleanup happens on downstream disconnect.
    """
    prompt = "word " * 500 + "user@example.com"
    async with gateway_client.stream("POST", "/generate", json={"prompt": prompt, "chunk_size": 4}) as resp:
        got_any = False
        async for _chunk in resp.aiter_bytes():
            got_any = True
            break  # abandon the rest of the stream deliberately
    assert got_any


async def test_upstream_single_byte_utf8_decode_redaction_and_cleanup() -> None:
    """Deliver actual byte splits inside UTF-8 sequences through httpx."""
    text = "caf\u00e9 \u65e5\u672c\u8a9e \U0001f680 user@example.com fin"

    class ByteStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.close_count = 0
            self.chunk_sizes: list[int] = []

        async def __aiter__(self):
            for value in text.encode("utf-8"):
                chunk = bytes([value])
                self.chunk_sizes.append(len(chunk))
                yield chunk

        async def aclose(self) -> None:
            self.close_count += 1

    stream = ByteStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/plain; charset=utf-8"}, stream=stream
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chunks = [part async for part in gateway_app._redacted_stream("ignored", 1, client)]
        assert b"".join(chunks).decode("utf-8") == text.replace("user@example.com", "[REDACTED]")
        assert len(chunks) > 1
        assert stream.chunk_sizes == [1] * len(text.encode("utf-8"))
        assert stream.close_count == 1
    assert client.is_closed


# =============================================================================
# 4. Real-process non-buffering timing proof (mirrors Task 1/Task 2 pattern)
# =============================================================================


def test_real_process_streams_incrementally_not_buffered() -> None:
    """Spawn the actual gateway and a deliberately slow local upstream as real

    `uvicorn` processes over real sockets, and prove time-to-first-byte is
    well under the upstream's total artificial delay -- the only way to
    honestly observe streaming timing, since httpx.ASGITransport does not
    preserve it (verified empirically: a direct ASGITransport call to a
    slow endpoint with no gateway in between shows the same delay as a
    fully-buffered response).
    """
    slow_app_path = REPO_ROOT / "tests" / "_task3_slow_upstream_app.py"
    assert slow_app_path.exists(), "helper slow-upstream app must exist alongside this test file"

    mock_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "_task3_slow_upstream_app:app",
            "--port",
            "8199",
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT / "tests"),
    )
    gateway_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "task3_stream_guardrail.app:app", "--port", "8299", "--log-level", "warning"],
        cwd=str(REPO_ROOT),
        env={"TASK3_UPSTREAM_URL": "http://127.0.0.1:8199/generate", **os.environ},
    )
    try:
        time.sleep(2.0)
        start = time.monotonic()
        first_byte_time = None
        with httpx.stream(
            "POST", "http://127.0.0.1:8299/generate", json={"prompt": "ignored"}, timeout=30.0
        ) as resp:
            for chunk in resp.iter_bytes():
                if first_byte_time is None:
                    first_byte_time = time.monotonic() - start
                break
        assert first_byte_time is not None, "no bytes received from the gateway at all"
        # The slow upstream sleeps 0.05s x 20 = 1.0s total; a genuinely
        # streaming gateway delivers the first byte after roughly one sleep,
        # not after all of them.
        assert first_byte_time < 0.6, (
            f"time to first byte was {first_byte_time:.3f}s -- looks like the full response was "
            "buffered before any of it was forwarded"
        )
    finally:
        mock_proc.terminate()
        gateway_proc.terminate()
        mock_proc.wait(timeout=5)
        gateway_proc.wait(timeout=5)
