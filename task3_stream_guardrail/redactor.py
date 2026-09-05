r"""Bounded, candidate-aware streaming PII redactor.

Email candidates retain at most 254 characters, numeric candidates at most
37, and short overlapping numeric chains at most 111. These limits derive
from the supported grammars; ordinary prose is released at its candidate
boundary. A large incoming chunk is scanned before its bounded suffix is
retained.

When a long chain must be confirmed, we emit only up to the earliest possible
future candidate start. The raw suffix stays available for overlap discovery.
An integer records how much of that suffix is already covered by an emitted
marker. Only a genuinely overlapping detected span can extend that coverage;
touching spans and intervening spaces/hyphens remain independent.

Card discovery checks whole blocks, real-left-boundary bare prefixes, and
recognized 4-4-4-4 / 4-6-5 layouts. The preceding-digit bit prevents a trim
from inventing a bare-card boundary. All card paths retain the Luhn gate.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# --- Pattern-derived bounds --------------------------------------------------
# These are NOT arbitrary constants: each is derived from the format it bounds,
# and documented as such, per the requirement to avoid a simplistic fixed-size
# trailing buffer.

# Practical SMTP mailbox limit: 256-octet path minus '<' and '>' (RFC 5321).
# Supported addresses are ASCII, so characters and octets have equal lengths.
EMAIL_LOCAL_MAX = 64
EMAIL_LABEL_MAX = 63
EMAIL_TOTAL_MAX = 254

# SSN: only the fixed "DDD-DD-DDDD" format is supported (per assessment
# wording "standard SSN form such as 123-45-6789") -- exactly 11 characters.
SSN_MAX = 11

# Credit card: 13-19 digits (the range Luhn-checkable card schemes use),
# each pair of digits joined by at most one separator -- worst case 19
# digits + 18 one-character separators = 37 characters.
CREDIT_CARD_MIN_DIGITS = 13
CREDIT_CARD_MAX_DIGITS = 19
CREDIT_CARD_MAX = 37

NUMERIC_CANDIDATE_MAX = max(SSN_MAX, CREDIT_CARD_MAX)

# Preserve the existing three-card-slot natural-closure window for short
# numeric chains. Longer chains commit a prefix while retaining every possible
# future candidate start and the exact remaining confirmed coverage.
MAX_MERGE_HOLDBACK = NUMERIC_CANDIDATE_MAX * 3

# --- Patterns -----------------------------------------------------------
# Applied only to text already proven safe (see algorithm above), so no
# lookahead trickery is needed here to guard against "might still extend" --
# that ambiguity is resolved entirely by the candidate-boundary computation.

EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,124}\.[A-Za-z]{2,24}"
)
_EMAIL_LOCAL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-")


def _email_local_starts(text: str, at: int) -> range:
    """Every plausible local-part start for this '@', in a bounded window."""
    start = at
    while start > 0 and at - start < EMAIL_LOCAL_MAX:
        if text[start - 1] not in _EMAIL_LOCAL_CHARS:
            break
        start -= 1
    return range(start, at)


def _find_email_spans(text: str) -> list[tuple[int, int]]:
    """Evaluate all local starts, each with its own 254-character budget.

    Preceding PII may be valid local-part characters. Its longer reading must
    not mask a shorter local part whose domain fits the total address limit.
    Only the longest end for a given start is needed: shorter readings at the
    same start are contained in that span. Distinct starts remain independent
    until overlap union. There are at most 64 starts per '@'.
    """
    spans = []
    for at in re.finditer("@", text):
        for start in _email_local_starts(text, at.start()):
            match = EMAIL_PATTERN.match(text, start, start + EMAIL_TOTAL_MAX)
            if match:
                spans.append(match.span())
    return spans


# Zero-width lookahead visits every start, including those inside another SSN.
# Capture 1 supplies the full canonical span; the outer match consumes nothing.
SSN_PATTERN = re.compile(r"(?=(\d{3}-\d{2}-\d{4}))")

# A *maximal* run of digits with at most one space/hyphen separator between
# any two consecutive digits -- deliberately unbounded in length and with no
# Luhn/length filtering. `_find_card_spans` below decomposes each such run
# into digit "blocks" (split at the separators) and tests every contiguous
# combination of whole blocks, so it is the superset every real card
# candidate must live inside, not a competing match against them.
_NUMERIC_RUN_PATTERN = re.compile(r"\d(?:[ -]?\d)*")
_DIGIT_BLOCK_PATTERN = re.compile(r"\d+")

# The 4-4-4-4 and 4-6-5 layouts identify both edge groups even when bare
# digits have joined them. Other separator layouts retain whole-block
# detection; we do not invent arbitrary partial groups. Lookahead keeps
# overlapping formatted candidates discoverable, including at buffer index 0.
_GROUPED_CARD_PATTERN = re.compile(
    r"(?=(\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}"
    r"|\d{4}[ -]\d{6}[ -]\d{5}))"
)

_SEPARATORS = re.compile(r"[ -]")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0 and len(digits) > 0


def _is_real_card(candidate: str) -> bool:
    digits = _SEPARATORS.sub("", candidate)
    return CREDIT_CARD_MIN_DIGITS <= len(digits) <= CREDIT_CARD_MAX_DIGITS and _luhn_valid(digits)


def _find_card_spans(text: str, *, suppress_start: bool = False) -> list[tuple[int, int]]:
    """Every valid (13-19 digit, Luhn-valid) card span in `text`.

    A single greedy regex match per numeric run is not enough: the longest
    digit-shaped candidate at a given start can fail Luhn while a *shorter or
    differently-positioned* candidate overlapping or adjacent to it is a real
    card (e.g. "123-45-6789 4111 1111 1111 1111" -- the greedy 19-digit
    candidate spanning the SSN and the start of the card fails Luhn, and a
    plain `finditer` resumes scanning after it, skipping over the valid card
    entirely).

    Instead, each maximal digit/separator run is decomposed into digit
    "blocks" split at the (single, optional) space/hyphen separators.
    Every contiguous combination
    of *whole* blocks whose digit count falls in [13, 19] is tried and every
    one that passes Luhn is returned (not just the first found): two
    genuinely different valid readings of the same digits can coexist (a
    Luhn-valid card that happens to overlap a syntactically valid SSN, or two
    adjacent valid cards whose boundary rotation also happens to validate),
    and `_pii_spans` is what safely reconciles overlapping results by taking
    their union rather than arbitrarily preferring one. Since digit counts
    only grow as more blocks are added, the inner loop stops as soon as a
    combination exceeds 19 digits, keeping this linear in the number of
    blocks rather than quadratic.

    Additionally, each bare block's real left boundary anchors 13-19 digit
    prefixes, so trailing digits cannot hide a valid card. We never scan
    interior/suffix windows of unformatted blocks. Recognized grouped
    layouts independently identify both edges despite adjacent bare digits.
    All three discovery paths retain the same length and Luhn gate.
    """
    spans: list[tuple[int, int]] = []
    for run in _NUMERIC_RUN_PATTERN.finditer(text):
        blocks = [(m.start(), m.end()) for m in _DIGIT_BLOCK_PATTERN.finditer(run.group())]
        run_start = run.start()
        blocks = [(run_start + s, run_start + e) for s, e in blocks]
        prefix = [0]
        for s, e in blocks:
            prefix.append(prefix[-1] + (e - s))
        for i in range(len(blocks)):
            if suppress_start and blocks[i][0] == 0:
                continue
            for j in range(i, len(blocks)):
                total_digits = prefix[j + 1] - prefix[i]
                if total_digits > CREDIT_CARD_MAX_DIGITS:
                    break
                if total_digits < CREDIT_CARD_MIN_DIGITS:
                    continue
                start, end = blocks[i][0], blocks[j][1]
                if _is_real_card(text[start:end]):
                    spans.append((start, end))
            # A bare run has a known left edge, but no evidence for an
            # interior start. Only check bounded prefixes at that edge.
            start, block_end = blocks[i]
            for end in range(start + CREDIT_CARD_MIN_DIGITS, min(block_end, start + CREDIT_CARD_MAX_DIGITS) + 1):
                if _is_real_card(text[start:end]):
                    spans.append((start, end))
    for match in _GROUPED_CARD_PATTERN.finditer(text):
        if _is_real_card(match.group(1)):
            spans.append(match.span(1))
    return spans


def _numeric_tail_open(text: str, end: int, n: int) -> bool:
    """True if every character from `end` to `n` belongs to the digit/space/

    hyphen candidate alphabet -- i.e. nothing between a span's own end and
    the end of currently-known text rules out the possibility that more
    digits, arriving later, would reveal a different, equally valid card
    window chaining onto it here. Using the *general* candidate alphabet
    (not `_NUMERIC_RUN_PATTERN`, which only matches digit-anchored runs) is
    what matters: a run that currently ends in a trailing space still counts
    as open, since a digit arriving right after that space would extend it
    exactly the way `_numeric_candidate_start` already treats space as part
    of the same alphabet.
    """
    for i in range(end, n):
        c = text[i]
        if not (c.isdigit() or c in " -"):
            return False
    return True


def _raw_pii_spans(text: str, *, suppress_start: bool = False) -> list[tuple[int, int]]:
    """Every individual (start, end) PII match in `text`, sorted by start,

    *before* overlap resolution. Each pattern is scanned independently
    rather than through one alternation: a card-shaped numeric region that
    fails Luhn contributes no span at all, so a genuine SSN sitting inside
    that region is still found on its own (a single alternation would have
    consumed the whole region as a "card", and returning it unchanged on
    Luhn failure silently suppressed the SSN -- e.g. "123-45-6789 1234"
    leaked the SSN before an earlier fix).

    `suppress_start`, when true, discards boundary-dependent card matches at
    index 0, but not independently structured SSNs or grouped cards.
    That position is special: after `StreamingRedactor` trims its
    buffer, position 0 of the new buffer may actually sit mid-digit-run in
    the true logical stream (the digits before it were already emitted), so
    a lookbehind like `(?<!\\d)` there is satisfied *vacuously* rather than
    because a real non-digit precedes it. Every other position is unaffected,
    since the character before it is always genuinely present in `text`. See
    the module docstring for the full argument.

    Each individual span is independently bounded (<=254 chars for email,
    <=37 for cards, 11 for SSNs). The streaming frontier preserves every
    still-possible start; committed overlap coverage survives subsequent trims.
    """
    raw: list[tuple[int, int]] = []
    raw.extend(_find_email_spans(text))
    for match in SSN_PATTERN.finditer(text):
        raw.append(match.span(1))
    for start, end in _find_card_spans(text, suppress_start=suppress_start):
        raw.append((start, end))
    raw.sort()
    return raw


def _merge_spans(raw: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union overlapping (start, end) spans from a sorted `raw` list.

    Two legitimately different valid readings of the same digits (e.g. a
    Luhn-valid card that happens to overlap a syntactically valid SSN, or two
    adjacent valid cards whose shared boundary rotation also happens to
    validate) must both be covered, since discarding either risks leaving a
    raw fragment of it exposed. Spans that only touch (one's end equals
    another's start) are not merged, so adjacent-but-distinct PII still
    renders as separate `[REDACTED]` markers. Note that a merged span's own
    *length* is not bounded the way an individual raw span's is: chaining
    enough overlapping windows together can make it arbitrarily long, which
    is exactly why `feed()` never uses this list for its cut-safety check.
    """
    if not raw:
        return []
    merged: list[tuple[int, int]] = [raw[0]]
    for start, end in raw[1:]:
        last_start, last_end = merged[-1]
        if start < last_end:
            if end > last_end:
                merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def _pii_spans(text: str, *, suppress_start: bool = False) -> list[tuple[int, int]]:
    """Accepted (start, end) spans of PII in `text`, merged where they overlap.

    Convenience wrapper for callers (like `flush()`) that only need the final
    answer with complete information already in hand -- see `_raw_pii_spans`
    and `_merge_spans` for what each step does and why `feed()` needs them
    kept separate.
    """
    return _merge_spans(_raw_pii_spans(text, suppress_start=suppress_start))


def _email_candidate_start(s: str) -> int:
    """Earliest still-possible email start, within explicit grammar bounds.

    The suffix can always become a new local part. An earlier '@' extends
    holdback only while its domain labels and total address still fit.
    """
    n = len(s)
    local_start = n
    while local_start > 0 and n - local_start < EMAIL_LOCAL_MAX:
        if s[local_start - 1] not in _EMAIL_LOCAL_CHARS:
            break
        local_start -= 1

    at = s.rfind("@", max(0, n - EMAIL_TOTAL_MAX))
    if at < 0:
        return local_start
    starts = _email_local_starts(s, at)
    # Expiring an earlier reading must not discard shorter local starts that
    # can still form a full address. Retain the earliest viable one, not just
    # the originally longest local part. This also caps all email state at 254.
    start = max(starts.start, n - EMAIL_TOTAL_MAX)
    if start >= at:
        return local_start
    labels = s[at + 1:].split(".")
    if any(len(label) > EMAIL_LABEL_MAX or any(
        not (ch.isascii() and (ch.isalnum() or ch == "-")) for ch in label
    ) for label in labels):
        return local_start
    if any(not label for label in labels[:-1]):
        return local_start
    return min(start, local_start)


def _numeric_candidate_start(s: str) -> int:
    """Leftmost index at/after which `s`'s suffix could still be forming an SSN or card."""
    i = len(s)
    consumed = 0
    while i > 0:
        c = s[i - 1]
        if not (c.isdigit() or c in " -"):
            break
        if consumed + 1 > NUMERIC_CANDIDATE_MAX:
            break
        consumed += 1
        i -= 1
    return i


class StreamingRedactor:
    """One response stream; retained raw state is at most EMAIL_TOTAL_MAX."""

    __slots__ = ("_pending", "_unsafe_start", "_covered_until")

    def __init__(self) -> None:
        self._pending = ""
        self._unsafe_start = False
        # Prefix of pending already covered by a marker emitted previously.
        # Its original start is strictly left of pending index 0.
        self._covered_until = 0

    def feed(self, chunk: str) -> str:
        """Return resolved text, retaining raw overlap context for future PII."""
        self._pending += chunk
        pending = self._pending
        n = len(pending)
        frontier = min(_email_candidate_start(pending), _numeric_candidate_start(pending))
        raw = _raw_pii_spans(pending, suppress_start=self._unsafe_start)

        # Keep the natural-closure behavior for short numeric chains. Once the
        # chain exceeds three card slots, commit at the candidate frontier,
        # NOT at the newest span's end: future overlapping windows can start
        # anywhere in the retained suffix.
        for start, end in _merge_spans(raw):
            if start < frontier and n - start <= MAX_MERGE_HOLDBACK:
                if _numeric_tail_open(pending, end, n):
                    frontier = start
                    break

        # Starts before the frontier cannot acquire new interpretations from
        # future input. Later starts remain unresolved even if they currently
        # happen to match (e.g. a provisional whole-block card at chunk EOF).
        stable = [(start, end) for start, end in raw if start < frontier]
        return self._commit(frontier, stable)

    def _commit(self, upto: int, spans: list[tuple[int, int]]) -> str:
        if self._covered_until:
            spans = [(-1, self._covered_until), *spans]
        merged = _merge_spans(sorted(spans))
        out = []
        cursor = 0
        covered_until = 0
        for start, end in merged:
            if start >= upto:
                break
            out.append(self._pending[cursor:max(cursor, start)])
            if start >= 0:
                out.append(REDACTED)
            cursor = min(end, upto)
            if end > upto:
                covered_until = end - upto
        out.append(self._pending[cursor:upto])
        if upto:
            self._unsafe_start = self._pending[upto - 1].isdigit()
        self._pending = self._pending[upto:]
        self._covered_until = covered_until
        return "".join(out)

    def flush(self) -> str:
        """Resolve EOF and reset all stream-specific context."""
        spans = _raw_pii_spans(self._pending, suppress_start=self._unsafe_start)
        out = self._commit(len(self._pending), spans)
        self._unsafe_start = False
        self._covered_until = 0
        return out

    @property
    def pending_size(self) -> int:
        """All retained raw text, including overlap already covered by a marker."""
        return len(self._pending)
