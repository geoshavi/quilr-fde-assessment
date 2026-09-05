"""Digit-adjacency regressions and an injected-ground-truth streaming oracle.

Run a fresh corpus with TASK3_FUZZ_SEED and TASK3_FUZZ_CASES. No production
matcher, span finder, or Luhn function is used to construct the oracle.
"""

from __future__ import annotations

import os
import random

import pytest

from task3_stream_guardrail.redactor import EMAIL_TOTAL_MAX, MAX_MERGE_HOLDBACK, StreamingRedactor

MARKER = "[REDACTED]"


def run(chunks: list[str]) -> tuple[str, int]:
    redactor = StreamingRedactor()
    parts = []
    peak = 0
    for chunk in chunks:
        parts.append(redactor.feed(chunk))
        peak = max(peak, redactor.pending_size)
        assert redactor.pending_size <= EMAIL_TOTAL_MAX
    parts.append(redactor.flush())
    assert redactor.pending_size == 0
    return "".join(parts), peak


CASES = [
    ("4111 1111 1111 11119", "[REDACTED]9"),
    ("41111111111111119", "[REDACTED]9"),
    ("123-45-67899", "[REDACTED]9"),
    ("123-45-67894111 1111 1111 1111", "[REDACTED]"),
    ("94111 1111 1111 1111", "9[REDACTED]"),
    ("94111-1111-1111-11119", "9[REDACTED]9"),
    ("9123-45-67899", "9[REDACTED]9"),
    ("93782 822463 100059", "9[REDACTED]9"),
    ("4111111111111112", "4111111111111112"),
    ("1234567890123456", "1234567890123456"),
    ("1234 5678 9012 3456", "1234 5678 9012 3456"),
    ("94111111111111111", "94111111111111111"),
    ("9" * 400, "9" * 400),
    ("9" * 400 + "4111111111111111", "9" * 400 + "4111111111111111"),
    ("9" * 400 + "123-45-6789" + "9" * 400, "9" * 400 + MARKER + "9" * 400),
    ("9" * 400 + "4111 1111 1111 1111" + "9" * 400, "9" * 400 + MARKER + "9" * 400),
    ("4111111111111111" + "9" * 400, MARKER + "9" * 400),
    ("123-45-6789" + "9" * 150 + "!user@example.com!", MARKER + "9" * 150 + "!" + MARKER + "!"),
]


@pytest.mark.parametrize("text,expected", CASES, ids=[f"adjacency-{index}" for index in range(len(CASES))])
def test_bare_digit_adjacency(text: str, expected: str) -> None:
    strategies = [[text], list(text)]
    strategies += [[text[:cut], "", text[cut:]] for cut in range(len(text) + 1)]
    strategies += [[text[i:i + size] for i in range(0, len(text), size)] for size in (2, 7, 37, 64, 111)]
    for chunks in strategies:
        output, _ = run(chunks)
        assert output == expected, f"sizes={[len(c) for c in chunks]}, output={output!r}"


def covers_injections(text: str, output: str, injected: list[tuple[int, int]]) -> bool:
    """Align visible output to source; only markers may consume injected bytes.

    A marker may also absorb neighboring noise or overlapping valid readings.
    This checks *all characters* of each injection, not just absence of the
    whole secret, so a partial leak fails too. No discovered production spans
    participate in the expected result.
    """
    protected = [False] * len(text)
    for start, end in injected:
        protected[start:end] = [True] * (end - start)
    positions = {0}
    cursor = 0
    while cursor < len(output) and positions:
        if output.startswith(MARKER, cursor):
            positions = set(range(min(positions) + 1, len(text) + 1))
            cursor += len(MARKER)
        else:
            positions = {
                pos + 1 for pos in positions
                if pos < len(text) and not protected[pos] and text[pos] == output[cursor]
            }
            cursor += 1
    return cursor == len(output) and len(text) in positions


def test_ground_truth_oracle_rejects_partial_leaks() -> None:
    text = "pre!123-45-6789!post"
    spans = [(4, 15)]
    assert covers_injections(text, "pre![REDACTED]!post", spans)
    assert covers_injections(text, "pre[REDACTED]post", spans)
    assert not covers_injections(text, text, spans)
    assert not covers_injections(text, "pre!123-45-[REDACTED]!post", spans)
    assert not covers_injections(text, "pre![REDACTED]6789!post", spans)


def test_adjacency_pending_state_stress() -> None:
    corpus = [
        "9" * 10000 + "4111 1111 1111 1111" + "9" * 10000,
        "9" * 10000 + "123-45-6789" + "9" * 10000,
        "4111111111111111" + "9" * 20000,
        "4111 1111 1111 1111 " * 250,
        "a@" + "b" * 5000,
        "a@" + ".".join(["b" * 60] * 20) + ".com",
    ]
    peak = 0
    for text in corpus:
        baseline, pending = run([text])
        peak = max(peak, pending)
        for size in (1, 37, 511):
            output, pending = run([text[i:i + size] for i in range(0, len(text), size)])
            peak = max(peak, pending)
            assert output == baseline
    assert peak == EMAIL_TOTAL_MAX
    print(f"pending-state stress: max_pending={peak}")


def make_card(rng: random.Random, length: int) -> str:
    """Generate a check digit using a left-to-right doubling table."""
    body = [rng.randrange(10) for _ in range(length - 1)]
    body[0] = rng.randrange(1, 10)
    doubled = (0, 2, 4, 6, 8, 1, 3, 5, 7, 9)
    total = sum(doubled[digit] if index % 2 == length % 2 else digit for index, digit in enumerate(body))
    return "".join(map(str, body)) + str((-total) % 10)


def test_injected_span_fuzz() -> None:
    seed = int(os.environ.get("TASK3_FUZZ_SEED", "20260905"))
    count = int(os.environ.get("TASK3_FUZZ_CASES", "300"))
    rng = random.Random(seed)
    peak = 0
    streams = 0
    for case in range(count):
        text = "begin!"
        injected = []
        for _ in range(rng.randint(1, 4)):
            kind = rng.choice(("raw", "grouped", "amex", "ssn", "pair", "ssn_chain", "email_overlap"))
            left = "".join(str(rng.randrange(10)) for _ in range(rng.choice((0, 1, 2, 7, 45, 150))))
            right = "".join(str(rng.randrange(10)) for _ in range(rng.choice((1, 2, 5, 40, 150))))
            if kind == "raw":
                # The policy requires a real left boundary for unformatted PII.
                left = ""
                secret = make_card(rng, rng.randint(13, 19))
            elif kind == "ssn":
                digits = "".join(str(rng.randrange(10)) for _ in range(9))
                secret = f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
            elif kind == "ssn_chain":
                secret, _ = overlapping_ssn_chain(rng.randint(2, 20), rng)
            elif kind == "email_overlap":
                # Every character is injected PII, including the overlapping
                # SSN chain. A raw card prefix still needs its real left edge.
                left = ""
                prefix = make_card(rng, rng.randint(13, 19)) if rng.randrange(2) else overlapping_ssn_chain(rng.randint(1, 4), rng)[0]
                suffix = make_card(rng, rng.randint(13, 19)) if rng.randrange(2) else overlapping_ssn_chain(rng.randint(1, 4), rng)[0]
                secret = prefix + max_length_email(rng.randint(1, 64)) + suffix
            else:
                digits = make_card(rng, 15 if kind == "amex" else 16)
                groups = (digits[:4], digits[4:10], digits[10:]) if kind == "amex" else tuple(digits[i:i + 4] for i in range(0, 16, 4))
                secret = rng.choice((" ", "-")).join(groups)
                if kind == "pair":
                    ssn_digits = "".join(str(rng.randrange(10)) for _ in range(9))
                    ssn = f"{ssn_digits[:3]}-{ssn_digits[3:5]}-{ssn_digits[5:]}"
                    secret = ssn + secret if rng.randrange(2) else secret + ssn
            text += left
            start = len(text)
            text += secret
            injected.append((start, len(text)))
            text += right + rng.choice(("!", "? safe!", "!user@example.com!"))
        text += "end!"
        strategies = [[text], list(text)]
        for _ in range(3):
            chunks = []
            pos = 0
            while pos < len(text):
                size = rng.randint(1, 80)
                chunks.append(text[pos:pos + size])
                pos += size
            strategies.append(chunks)
        baseline = None
        for chunks in strategies:
            output, pending = run(chunks)
            peak = max(peak, pending)
            streams += 1
            context = f"seed={seed}, case={case}, text={text!r}, injected={injected}, chunks={chunks!r}, output={output!r}"
            assert covers_injections(text, output, injected), context
            if baseline is None:
                baseline = output
            assert output == baseline, context
            assert output.startswith("begin!") and output.endswith("end!"), context
    print(f"injected-span fuzz: seed={seed}, cases={count}, streams={streams}, max_pending={peak}")


def regression_chunkings(text: str) -> list[list[str]]:
    return [[text], list(text)] + [
        [text[i:i + size] for i in range(0, len(text), size)]
        for size in (2, 7, 37, 40, 64, 111, 254)
    ]


@pytest.mark.parametrize("repeats", range(2, 11))
@pytest.mark.parametrize("suffix,expected", [
    ("", MARKER),
    (" 123-45-6789", MARKER + " " + MARKER),
    (" user@example.com", MARKER + " " + MARKER),
])
def test_repeated_chain_retains_overlap_context(repeats: int, suffix: str, expected: str) -> None:
    card = "4111 1111 1111 1111"
    text = " ".join([card] * repeats) + suffix
    injections = [(i * 20, i * 20 + len(card)) for i in range(repeats)]
    for chunks in regression_chunkings(text):
        output, _ = run(chunks)
        assert covers_injections(text, output, injections), (repeats, chunks, output)
        assert output == expected, (repeats, chunks, output)
    # The oracle must reject a partial leak even though the complete secret
    # does not survive. No production matcher supplies these source offsets.
    assert not covers_injections(text, MARKER + " 1111 1111" + suffix, injections)


@pytest.mark.parametrize("repeats", range(2, 11))
@pytest.mark.parametrize("layout", ["4-4-4-4", "4-6-5"])
def test_generated_repeated_chains_with_independent_injections(repeats: int, layout: str) -> None:
    rng = random.Random(927100 + repeats)
    digits = make_card(rng, 16 if layout == "4-4-4-4" else 15)
    groups = [digits[i:i + 4] for i in range(0, 16, 4)] if len(digits) == 16 else [digits[:4], digits[4:10], digits[10:]]
    card = "-".join(groups)
    text = " ".join([card] * repeats)
    injections = [(i * (len(card) + 1), i * (len(card) + 1) + len(card)) for i in range(repeats)]
    baseline = None
    for chunks in regression_chunkings(text):
        output, _ = run(chunks)
        assert covers_injections(text, output, injections), (text, chunks, output)
        if baseline is None:
            baseline = output
        assert output == baseline, (text, chunks, output)


@pytest.mark.parametrize("left,right", [
    ("123-45-6789", "4111111111111111"),
    ("4111111111111111", "123-45-6789"),
    ("4111111111111111", "4111111111111111"),
])
@pytest.mark.parametrize("length", [2, 37, 40, MAX_MERGE_HOLDBACK - 1, MAX_MERGE_HOLDBACK, MAX_MERGE_HOLDBACK + 1, 120, 254, 1000])
@pytest.mark.parametrize("separator", [" ", "-"])
def test_independent_pii_preserves_all_separators(left: str, right: str, length: int, separator: str) -> None:
    gap = separator * length
    text = left + gap + right
    for chunks in regression_chunkings(text):
        output, _ = run(chunks)
        assert output == MARKER + gap + MARKER, (chunks, output)


@pytest.mark.parametrize("text", [
    "a" * 5000 + "@example.com!",
    "a@" + "b" * 5000 + ".com!",
    "a@" + ".".join(["b" * 60] * 60) + ".com!",
    ("a@b.." * 1000) + "!",
    ("a@-._" * 1000) + "!",
])
def test_malformed_email_streams_release_bounded_state(text: str) -> None:
    baseline = None
    for chunks in regression_chunkings(text):
        output, peak = run(chunks)
        assert peak <= EMAIL_TOTAL_MAX
        if baseline is None:
            baseline = output
        assert output == baseline
    redactor = StreamingRedactor()
    emitted = "".join(redactor.feed(ch) for ch in text[:-1])
    assert emitted, "overlong candidates must release text before EOF"


@pytest.mark.parametrize("email", [
    "user@example.com",
    "a" * 64 + "@example.com",
    "user@" + "b" * 63 + ".com",
    "a@" + ".".join(["b" * 60] * 4) + ".com",  # 249 characters
    "a" * 64 + "@" + ".".join(["b" * 61] * 3) + ".com",  # exactly 254
])
def test_bounded_valid_emails_at_every_split(email: str) -> None:
    assert len(email) <= EMAIL_TOTAL_MAX
    text = "before!" + email + "!after"
    strategies = regression_chunkings(text) + [[text[:i], "", text[i:]] for i in range(len(text) + 1)]
    for chunks in strategies:
        output, _ = run(chunks)
        assert output == "before!" + MARKER + "!after", (chunks, output)


def overlapping_ssn_chain(count: int, rng: random.Random) -> tuple[str, list[tuple[int, int]]]:
    """Construct canonical SSNs sharing three digits; record before matching."""
    def digits(size: int) -> str:
        return "".join(str(rng.randrange(10)) for _ in range(size))
    text = digits(3) + "-" + digits(2) + "-" + digits(4)
    injections = [(0, len(text))]
    for _ in range(count - 1):
        start = len(text) - 3
        text += "-" + digits(2) + "-" + digits(4)
        injections.append((start, len(text)))
    return text, injections


@pytest.mark.parametrize("count", [2, 3, 6, 10, 20, 100])
def test_overlapping_ssn_starts(count: int) -> None:
    text, injections = overlapping_ssn_chain(count, random.Random(281 + count))
    for chunks in regression_chunkings(text):
        output, _ = run(chunks)
        assert covers_injections(text, output, injections), (chunks, output)
        assert output == MARKER


def test_reported_overlapping_ssn_starts() -> None:
    text = "123-45-6789-12-3456"
    injections = [(0, 11), (8, 19)]
    assert not covers_injections(text, MARKER + "-12-3456", injections)
    for chunks in regression_chunkings(text):
        output, _ = run(chunks)
        assert covers_injections(text, output, injections)
        assert output == MARKER


def max_length_email(local_size: int) -> str:
    """Construct exactly 254 ASCII characters with legal individual labels."""
    domain_budget = 254 - local_size - 1 - len(".com")
    labels = []
    while domain_budget > 63:
        width = min(63, domain_budget - 2)  # leave a nonempty final label
        labels.append("b" * width)
        domain_budget -= width + 1
    labels.append("b" * domain_budget)
    email = "a" * local_size + "@" + ".".join(labels) + ".com"
    assert 1 <= local_size <= 64
    assert all(1 <= len(label) <= 63 for label in labels)
    assert len(email) == 254
    return email


@pytest.mark.parametrize("local_size", [1, 2, 17, 32, 57, 63, 64])
@pytest.mark.parametrize("prefix,suffix", [
    ("", ""),
    ("4111111111111111", ""),
    ("123-45-6789", ""),
    ("", "4111111111111111"),
    ("", "123-45-6789"),
    ("4111111111111111", "123-45-6789"),
    ("123-45-6789", "4111111111111111"),
])
def test_max_length_email_all_local_starts(local_size: int, prefix: str, suffix: str) -> None:
    email = max_length_email(local_size)
    assert len(email) == 254
    text = prefix + email + suffix
    injections = [(len(prefix), len(prefix) + len(email))]
    if prefix:
        injections.append((0, len(prefix)))
    if suffix:
        injections.append((len(prefix) + len(email), len(text)))
    # A full 64-character local part cannot include any preceding PII.
    expected = MARKER * (1 + bool(suffix) + bool(prefix and local_size == 64))
    strategies = regression_chunkings(text) + [[text[:i], text[i:]] for i in range(len(text) + 1)]
    for chunks in strategies:
        output, _ = run(chunks)
        assert covers_injections(text, output, injections), (chunks, output)
        assert output == expected, (chunks, output)


@pytest.mark.parametrize("repeats", [2, 6, 10])
@pytest.mark.parametrize("gap", ["", " " * 120])
def test_overlapping_card_ssn_email_injections(repeats: int, gap: str) -> None:
    card = "4111 1111 1111 1111"
    prefix = " ".join([card] * repeats)
    ssns, ssn_injections = overlapping_ssn_chain(20, random.Random(802))
    email = max_length_email(1)
    text = prefix + gap + ssns + email + "!"
    injections = [(i * 20, i * 20 + len(card)) for i in range(repeats)]
    offset = len(prefix) + len(gap)
    injections += [(offset + start, offset + end) for start, end in ssn_injections]
    injections.append((offset + len(ssns), len(text) - 1))
    baseline = None
    for chunks in regression_chunkings(text):
        output, _ = run(chunks)
        assert covers_injections(text, output, injections), (chunks, output)
        if baseline is None:
            baseline = output
        assert output == baseline
        if gap:
            assert output == MARKER + gap + MARKER + "!"
