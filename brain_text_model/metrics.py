from __future__ import annotations

from typing import Iterable, Sequence


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, token_a in enumerate(a, start=1):
        curr[0] = i
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[-1]


def character_error_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    total_errors = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        errors = levenshtein(list(pred), list(ref))
        total_errors += errors
        total_chars += max(1, len(ref))
    return total_errors / total_chars if total_chars else 0.0


def _run_metrics_tests() -> None:
    """Test levenshtein and character_error_rate functions."""
    print("[metrics] Running tests...")
    
    # Test levenshtein distance
    assert levenshtein("", "") == 0
    assert levenshtein("a", "") == 1
    assert levenshtein("", "a") == 1
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("abc", "def") == 3
    assert levenshtein("kitten", "sitting") == 3
    
    # Test with sequences
    seq1 = list("hello")
    seq2 = list("hallo")
    assert levenshtein(seq1, seq2) == 1
    
    # Test character error rate
    predictions = ["hello", "world"]
    references = ["hello", "world"]
    cer = character_error_rate(predictions, references)
    assert cer == 0.0, f"Expected 0.0, got {cer}"
    
    # Test with errors
    predictions = ["helo", "wrld"]
    references = ["hello", "world"]
    cer = character_error_rate(predictions, references)
    # helo vs hello: 1 error / 5 chars = 0.2
    # wrld vs world: 1 error / 5 chars = 0.2
    # average: 2 / 10 = 0.2
    assert abs(cer - 0.2) < 1e-6, f"Expected 0.2, got {cer}"
    
    # Test with complete mismatch
    predictions = ["abc"]
    references = ["xyz"]
    cer = character_error_rate(predictions, references)
    assert cer == 1.0, f"Expected 1.0, got {cer}"
    
    # Test with empty prediction
    predictions = [""]
    references = ["hello"]
    cer = character_error_rate(predictions, references)
    assert cer == 1.0, f"Expected 1.0, got {cer}"
    
    # Test with empty lists
    cer = character_error_rate([], [])
    assert cer == 0.0
    
    # Test mixed cases
    predictions = ["hello world", "test", ""]
    references = ["hello world", "tset", "a"]
    cer = character_error_rate(predictions, references)
    # "hello world" vs "hello world": 0/11
    # "test" vs "tset": 2/4 = 0.5
    # "" vs "a": 1/1 = 1.0
    # total: 3 / 16 = 0.1875
    assert abs(cer - 0.1875) < 1e-6, f"Expected 0.1875, got {cer}"
    
    print("[metrics] ✓ All tests passed")


if __name__ == "__main__":
    _run_metrics_tests()
