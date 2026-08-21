"""Turn article text into the tokens BM25 counts.

This function defines the vocabulary of the whole retrieval system: two texts
can only ever match if it produces the same token for both. It must be
deterministic, and it must be applied identically to documents (offline) and
to queries (online).

Version A -- plain words. No stemming, no lemmatization. Those are arms B and
C, to be measured in Phase 8 rather than assumed here.
"""

from __future__ import annotations

import re

# A token is a run of ASCII letters or digits; everything else separates.
# Digits are kept deliberately -- "IPC 302", "Section 376" and years carry
# real meaning in a crime corpus.
TOKEN_PATTERN = re.compile(r"\w+")

# Dropped to shrink the index, NOT to improve ranking. BM25 already scores a
# word that appears in every document at essentially zero, so these cost
# memory without affecting results.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with",
})

# Single characters are almost always debris: the "s" left by "Shetty's",
# or a stray initial. They match everywhere and discriminate nothing.
MIN_TOKEN_LENGTH = 2

def tokenize(text: str) -> list[str]:
    """Return the tokens of `text`, in order of appearance.

    Order is preserved because it is free to keep and phrase search would
    need it later. BM25 itself ignores order entirely.
    """
    if not text:
        return []

    lowered = text.lower()
    return [
        token
        for token in TOKEN_PATTERN.findall(lowered)
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    ]

if __name__=="__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    samples = [
        "Ashram deaths: CID begins probe",
        "Inmate's pregnancy at Sasaram rocks jail dept",
        "Asaram Bapu was arrested in 2013 under IPC 376",
        '<div class="Normal"><br />MUMBAI: Boney Kapoor and his wife</div>',
    ]

    for sample in samples:
        print(f"\n{sample}")
        print(f" -> {tokenize(sample)}")