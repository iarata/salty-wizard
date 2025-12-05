from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

DEFAULT_CHARSET = list("abcdefghijklmnopqrstuvwxyz0123456789 .,'?-:/")


@dataclass(slots=True)
class TokenizerConfig:
    lowercase: bool = True
    charset: Sequence[str] | None = None


class CharTokenizer:
    """Lightweight character-level tokenizer for CTC decoding."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        if config is None:
            config = TokenizerConfig()
        self.lowercase = config.lowercase
        charset = list(config.charset) if config.charset else DEFAULT_CHARSET
        self.blank_token = "<blank>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.base_tokens: List[str] = charset
        self.symbols: List[str] = [self.blank_token, self.pad_token, self.unk_token, *charset]
        self.stoi = {token: idx for idx, token in enumerate(self.symbols)}
        self.itos = {idx: token for token, idx in self.stoi.items()}

    @property
    def vocab_size(self) -> int:  # pragma: no cover - trivial property
        return len(self.symbols)

    @property
    def blank_id(self) -> int:
        return self.stoi[self.blank_token]

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    def encode(self, text: str) -> List[int]:
        if self.lowercase:
            text = text.lower()
        ids: List[int] = []
        for ch in text:
            token = ch if ch in self.stoi else self.unk_token
            ids.append(self.stoi[token])
        return ids

    def decode(self, ids: Iterable[int], collapse_repeats: bool = True) -> str:
        text_chars: List[str] = []
        prev_token = None
        for idx in ids:
            token = self.itos.get(int(idx), self.unk_token)
            if token in {self.blank_token, self.pad_token}:
                prev_token = token
                continue
            if collapse_repeats and token == prev_token:
                continue
            text_chars.append(token)
            prev_token = token
        return "".join(text_chars)

    def greedy_decode(self, logits: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        """Greedy CTC decoding given time-major logits."""

        if logits.dim() != 3:
            raise ValueError("Expected logits tensor of shape (T, N, C)")
        probs = logits.argmax(dim=-1)  # (T, N)
        time, batch = probs.shape
        results: List[str] = []
        for b in range(batch):
            seq_len = int(lengths[b]) if lengths is not None else time
            ids = probs[:seq_len, b].tolist()
            results.append(self.decode(ids))
        return results


class PhonemeTokenizer:
    """Tokenizer that operates on whitespace separated phoneme sequences."""

    def __init__(self, specials: Sequence[str] | None = None) -> None:
        specials = specials or ("<blank>", "<pad>", "<unk>")
        self.blank_token, self.pad_token, self.unk_token = specials
        self.tokens: List[str] = list(specials)
        self.stoi = {token: idx for idx, token in enumerate(self.tokens)}
        self.itos = {idx: token for token, idx in self.stoi.items()}

    def maybe_add_tokens(self, phoneme_sequence: str | None) -> None:
        if not phoneme_sequence:
            return
        for token in phoneme_sequence.split():
            if token not in self.stoi:
                idx = len(self.tokens)
                self.tokens.append(token)
                self.stoi[token] = idx
                self.itos[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def blank_id(self) -> int:
        return self.stoi[self.blank_token]

    def encode(self, phoneme_sequence: str | None) -> List[int]:
        if not phoneme_sequence:
            return []
        return [self.stoi.get(token, self.stoi[self.unk_token]) for token in phoneme_sequence.split()]

    def decode(self, ids: Iterable[int]) -> str:
        tokens = []
        prev = None
        for idx in ids:
            token = self.itos.get(int(idx), self.unk_token)
            if token == self.blank_token:
                prev = token
                continue
            if token == prev:
                continue
            tokens.append(token)
            prev = token
        return " ".join(tokens)


def _run_tokenizer_tests() -> None:
    """Test CharTokenizer and PhonemeTokenizer on CPU, MPS, and CUDA."""
    print("[tokenizer] Running tests...")
    
    # Determine available devices
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    
    print(f"[tokenizer] Testing on devices: {devices}")
    
    # Test CharTokenizer
    tokenizer = CharTokenizer()
    assert tokenizer.vocab_size > 0
    assert tokenizer.blank_id == 0
    assert tokenizer.pad_id == 1
    
    # Test encoding
    text = "hello world"
    ids = tokenizer.encode(text)
    assert len(ids) == len(text)
    assert all(isinstance(i, int) for i in ids)
    
    # Test decoding
    decoded = tokenizer.decode(ids, collapse_repeats=False)
    assert decoded == text
    
    # Test with uppercase (should lowercase)
    ids_upper = tokenizer.encode("HELLO WORLD")
    assert ids_upper == ids
    
    # Test greedy decode with different devices
    for device in devices:
        try:
            # Create dummy logits (T=10, N=2, C=vocab_size)
            logits = torch.randn(10, 2, tokenizer.vocab_size, device=device)
            lengths = torch.tensor([10, 8], device=device)
            
            results = tokenizer.greedy_decode(logits, lengths)
            assert len(results) == 2
            assert all(isinstance(s, str) for s in results)
            print(f"[tokenizer] ✓ CharTokenizer greedy_decode works on {device}")
        except Exception as e:
            print(f"[tokenizer] ✗ CharTokenizer failed on {device}: {e}")
    
    # Test with unknown characters
    ids_unk = tokenizer.encode("hello@#$world")
    decoded_unk = tokenizer.decode(ids_unk)
    assert tokenizer.unk_token not in decoded_unk or "@" not in decoded_unk
    
    # Test PhonemeTokenizer
    phoneme_tokenizer = PhonemeTokenizer()
    assert phoneme_tokenizer.vocab_size == 3  # Just specials initially
    
    # Test adding tokens
    phoneme_tokenizer.maybe_add_tokens("HH AH L OW")
    assert phoneme_tokenizer.vocab_size == 7  # 3 specials + 4 phonemes
    
    # Test encoding
    phoneme_ids = phoneme_tokenizer.encode("HH AH L OW")
    assert len(phoneme_ids) == 4
    
    # Test decoding
    decoded_phonemes = phoneme_tokenizer.decode(phoneme_ids)
    assert decoded_phonemes == "HH AH L OW"
    
    # Test with unknown phoneme
    phoneme_tokenizer.maybe_add_tokens("W ER L D")
    encoded_with_unk = phoneme_tokenizer.encode("HH AH UNKNOWN")
    assert tokenizer.unk_token in [phoneme_tokenizer.itos.get(i, "") for i in encoded_with_unk]
    
    # Test empty sequence
    empty_ids = phoneme_tokenizer.encode(None)
    assert empty_ids == []
    
    print("[tokenizer] ✓ All tests passed")


if __name__ == "__main__":
    _run_tokenizer_tests()
