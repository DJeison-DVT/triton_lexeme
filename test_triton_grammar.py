"""
Red-green tests for the Triton kernel xgrammar validation.

- curated_100.jsonl   : valid kernels   -> grammar MUST accept   (should_parse=True)
- adversarial_100.jsonl : invalid kernels -> grammar MUST reject  (should_parse=False)

Run:
    pytest test_triton_grammar.py -v
"""

import json
import pathlib

import pytest

from triton_grammar import validate_code

DATA_DIR = pathlib.Path(__file__).parent

# ── fixtures ────────────────────────────────────────────────────────────────


def _load_jsonl(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


CURATED = _load_jsonl("curated_100.jsonl")
ADVERSARIAL = _load_jsonl("adversarial_100.jsonl")

# ── parameterised tests ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sample",
    CURATED,
    ids=[s.get("function_name", f"curated_{i}") for i, s in enumerate(CURATED)],
)
def test_curated_kernel_accepted(sample: dict):
    """Every curated (valid) kernel must be accepted by the grammar."""
    code = sample["kernel_code"]
    assert validate_code(code), (
        f"Grammar rejected valid kernel: {sample.get('function_name', '?')}"
    )


@pytest.mark.parametrize(
    "sample",
    ADVERSARIAL,
    ids=[
        f"{s.get('id', i)}[{s.get('category', '?')}/{s.get('subcategory', '?')}]"
        for i, s in enumerate(ADVERSARIAL)
    ],
)
def test_adversarial_kernel_rejected(sample: dict):
    """Every adversarial (invalid) kernel must be rejected by the grammar."""
    code = sample["kernel_code"]
    sid = sample.get("id", "?")
    cat = sample.get("category", "?")
    subcat = sample.get("subcategory", "?")
    desc = sample.get("error_description", "")
    assert not validate_code(code), (
        f"Grammar accepted invalid kernel {sid}\n"
        f"  category:    {cat}\n"
        f"  subcategory: {subcat}\n"
        f"  error:       {desc}\n"
        f"  code:        {code!r:.120}"
    )


# ── summary test ────────────────────────────────────────────────────────────


class TestGrammarSummary:
    """Aggregate pass-rate tests for quick feedback."""

    def test_curated_pass_rate(self):
        results = [validate_code(s["kernel_code"]) for s in CURATED]
        passed = sum(results)
        total = len(results)
        rate = passed / total * 100
        print(f"\nCurated pass rate: {passed}/{total} ({rate:.1f}%)")
        # Require 100% acceptance of valid kernels
        assert passed == total, f"Only {passed}/{total} curated kernels accepted"

    def test_adversarial_reject_rate(self):
        results = [(s, validate_code(s["kernel_code"])) for s in ADVERSARIAL]
        rejected = sum(not accepted for _, accepted in results)
        total = len(results)
        rate = rejected / total * 100

        # Build per-category breakdown
        from collections import defaultdict
        by_cat: dict[str, list] = defaultdict(list)
        for s, accepted in results:
            by_cat[s.get("category", "?")].append((s, accepted))

        lines = [f"\nAdversarial reject rate: {rejected}/{total} ({rate:.1f}%)"]
        for cat, entries in sorted(by_cat.items()):
            cat_rejected = sum(not a for _, a in entries)
            cat_total = len(entries)
            status = "OK" if cat_rejected == cat_total else "LEAK"
            lines.append(f"  [{status}] {cat}: {cat_rejected}/{cat_total}")
            if cat_rejected < cat_total:
                for s, accepted in entries:
                    if accepted:
                        lines.append(
                            f"        ^ {s.get('id')}: {s.get('subcategory')} "
                            f"- {s.get('error_description')}"
                        )
        print("\n".join(lines))

        assert rejected == total, (
            f"Only {rejected}/{total} adversarial kernels rejected. "
            f"See per-category breakdown above."
        )
