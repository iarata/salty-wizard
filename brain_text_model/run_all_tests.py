#!/usr/bin/env python
"""Run all brain_text_model tests."""

from __future__ import annotations

import sys


def main() -> int:
    """Run all tests and report results."""
    print("=" * 60)
    print("Running all brain_text_model tests")
    print("=" * 60)
    print()
    
    test_modules = [
        ("config", "Configuration dataclasses"),
        ("metrics", "Metrics (Levenshtein & CER)"),
        ("tokenizer", "CharTokenizer & PhonemeTokenizer"),
        ("data", "BrainTextDataModule"),
        ("model", "BrainToTextLightningModule"),
    ]
    
    failed = []
    
    for module_name, description in test_modules:
        print(f"\n{'=' * 60}")
        print(f"Testing: {description}")
        print('=' * 60)
        
        try:
            if module_name == "config":
                from .config import _run_config_tests
                _run_config_tests()
            elif module_name == "metrics":
                from .metrics import _run_metrics_tests
                _run_metrics_tests()
            elif module_name == "tokenizer":
                from .tokenizer import _run_tokenizer_tests
                _run_tokenizer_tests()
            elif module_name == "data":
                from .data import _run_smoke_test
                _run_smoke_test()
            elif module_name == "model":
                from .model import _run_model_tests
                _run_model_tests()
        except Exception as e:
            print(f"\n✗ {description} FAILED: {e}")
            failed.append((module_name, description, str(e)))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if failed:
        print(f"\n✗ {len(failed)} module(s) failed:")
        for module_name, description, error in failed:
            print(f"  - {description}: {error}")
        return 1
    else:
        print("\n✓ All tests passed!")
        print(f"  Tested {len(test_modules)} modules successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
