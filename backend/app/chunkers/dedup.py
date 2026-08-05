import difflib
from collections.abc import Sequence

from app.schemas.chunk import Chunk


def is_duplicate(
    new_chunk: Chunk,
    history: Sequence[Chunk],
    threshold: float,
) -> bool:
    """Return True if *new_chunk* is highly similar to any chunk in *history*.

    This performs intra-document string-based deduplication only.
    It does not address cross-corpus or vector-level duplication.

    Uses a fast-fail length check:
        The maximum possible SequenceMatcher ratio between two strings of
        lengths L1 and L2 is 2·min(L1,L2)/(L1+L2).
        If that maximum is already below *threshold*, the expensive comparison
        is skipped immediately.

    Args:
        new_chunk:  The candidate chunk to test.
        history:    Previously accepted chunks from the same document.
        threshold:  Minimum similarity ratio to consider a duplicate [0, 1].

    Returns:
        True if *new_chunk* should be considered a duplicate of any item in
        *history*, False otherwise.
    """
    if not history:
        return False

    l1 = len(new_chunk.content)

    for past in history:
        l2 = len(past.content)

        # Avoid zero-division; two empty strings are trivially "equal".
        if l1 == 0 and l2 == 0:
            return True

        total = l1 + l2
        max_possible = (2.0 * min(l1, l2) / total) if total else 1.0
        if max_possible < threshold:
            continue
        matcher = difflib.SequenceMatcher(None, new_chunk.content, past.content)

        # real_quick_ratio and quick_ratio are fast upper-bound estimates.
        if matcher.real_quick_ratio() < threshold:
            continue
        if matcher.quick_ratio() < threshold:
            continue
        if matcher.ratio() >= threshold:
            return True

    return False
