"""
test_scanner.py — Data-driven test harness for the Triton lexical analyzer.

All test cases live in test_cases.jsonl (one JSON object per line).
This file is a generic runner: add/modify/remove test cases by
editing the JSONL, not this Python code.

Curated and adversarial integration suites still load from the
parent project's JSONL datasets for full coverage.

Run:
    pytest test_scanner.py -v                       (all tests)
    pytest test_scanner.py -k "token_recognition"   (one category)
    pytest test_scanner.py -k "disambiguation"      (another)
    pytest test_scanner.py -k "curated" -v -s       (curated summary)
"""
import subprocess
import tempfile
import os
import re
import json
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCANNER_DIR  = os.path.dirname(os.path.abspath(__file__))
SCANNER_BIN  = os.path.join(SCANNER_DIR, "triton_scanner")
PROJECT_DIR  = os.path.dirname(SCANNER_DIR)
CASES_PATH   = os.path.join(SCANNER_DIR, "test_cases.jsonl")
CURATED_PATH = os.path.join(PROJECT_DIR, "curated_100.jsonl")
ADVERSARIAL_PATH = os.path.join(PROJECT_DIR, "adversarial_100.jsonl")


# ---------------------------------------------------------------------------
# Scanner runner
# ---------------------------------------------------------------------------
def run_scanner(code):
    """Write code to a temp file, run the scanner, return (stdout, stderr)."""
    fd, tmppath = tempfile.mkstemp(suffix=".triton", dir=SCANNER_DIR)
    try:
        with os.fdopen(fd, "w", newline="\n") as f:
            f.write(code)
        result = subprocess.run(
            [SCANNER_BIN, tmppath],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout, result.stderr
    finally:
        os.unlink(tmppath)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------
def parse_tokens(stdout):
    """Extract list of [token_type, lexeme, line] from scanner output."""
    tokens = []
    for line in stdout.splitlines():
        m = re.match(r"\s*\d+\s+<(\w+)\s*,\s*\"(.+)\",\s*(\d+)>", line)
        if m:
            tokens.append([m.group(1), m.group(2), int(m.group(3))])
    return tokens


def token_types(stdout):
    """Extract just the token type names."""
    return [t[0] for t in parse_tokens(stdout)]


def parse_table_entries(stdout, table_header):
    """Parse a symbol table section → list of (id, lexeme, first_line)."""
    lines = stdout.splitlines()
    entries = []
    in_section = False
    past_sep = False
    for line in lines:
        if table_header in line:
            in_section = True
            past_sep = False
            continue
        if in_section:
            if line.startswith("------"):
                past_sep = True
                continue
            if line.strip().startswith("ID"):
                continue
            if not line.strip() or "--- " in line or "===" in line:
                if past_sep:
                    break
                continue
            if past_sep:
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    entries.append((int(parts[0]), parts[1], int(parts[2])))
    return entries


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


TEST_CASES  = load_jsonl(CASES_PATH)
CURATED     = load_jsonl(CURATED_PATH)
ADVERSARIAL = load_jsonl(ADVERSARIAL_PATH)

SCANNER_DETECTABLE = {
    "invalid_token_dollar", "invalid_token_question",
    "unclosed_string_triple", "unclosed_brace",
}
ADVERSARIAL_SCANNER = [k for k in ADVERSARIAL
                       if k.get("category") == "lexer_errors"
                       and k.get("subcategory") in SCANNER_DETECTABLE]
ADVERSARIAL_PARSER  = [k for k in ADVERSARIAL
                       if k.get("category") == "lexer_errors"
                       and k.get("subcategory") not in SCANNER_DETECTABLE]


# ---------------------------------------------------------------------------
# Generic assertion engine
# ---------------------------------------------------------------------------
def check_case(case):
    """Run the scanner on case['input'] and verify all assertion fields.

    Supported fields (all optional — only checked if present):
        has_types           [str]   — token types that must appear
        not_types           [str]   — token types that must NOT appear
        exact_tokens        [[type, lexeme, line], ...]  — full match
        has_errors          bool    — True ⇒ stderr non-empty, False ⇒ empty
        min_tokens          int     — at least N tokens in output
        identifier_lexemes  [str]   — must appear in Identifier Table
        identifier_count    int     — exact row count of Identifier Table
        integer_lexemes     [str]   — must appear in Integer Constants Table
        float_lexemes       [str]   — must appear in Float Constants Table
        string_lexemes      [str]   — must appear in String Constants Table
    """
    stdout, stderr = run_scanner(case["input"])
    types = token_types(stdout)
    tokens = parse_tokens(stdout)
    cid = case["id"]

    # -- Token type presence --
    for t in case.get("has_types", []):
        assert t in types, f"[{cid}] expected token type {t} in output"

    # -- Token type absence --
    for t in case.get("not_types", []):
        assert t not in types, f"[{cid}] token type {t} should NOT appear"

    # -- Exact token sequence --
    if "exact_tokens" in case:
        assert tokens == case["exact_tokens"], (
            f"[{cid}] token mismatch:\n"
            f"  expected: {case['exact_tokens']}\n"
            f"  got:      {tokens}"
        )

    # -- Error expectation --
    if "has_errors" in case:
        if case["has_errors"]:
            assert stderr.strip() != "", f"[{cid}] expected errors on stderr"
        else:
            assert stderr.strip() == "", (
                f"[{cid}] unexpected errors:\n{stderr}"
            )

    # -- Minimum token count --
    if "min_tokens" in case:
        assert len(tokens) >= case["min_tokens"], (
            f"[{cid}] expected >= {case['min_tokens']} tokens, got {len(tokens)}"
        )

    # -- Symbol table: identifiers --
    if "identifier_lexemes" in case:
        entries = parse_table_entries(stdout, "Identifier Table")
        lexemes = [e[1] for e in entries]
        for lex in case["identifier_lexemes"]:
            assert lex in lexemes, (
                f"[{cid}] '{lex}' not found in Identifier Table"
            )

    if "identifier_count" in case:
        entries = parse_table_entries(stdout, "Identifier Table")
        assert len(entries) == case["identifier_count"], (
            f"[{cid}] Identifier Table has {len(entries)} entries, "
            f"expected {case['identifier_count']}"
        )

    # -- Symbol table: integers --
    if "integer_lexemes" in case:
        entries = parse_table_entries(stdout, "Integer Constants Table")
        lexemes = [e[1] for e in entries]
        for lex in case["integer_lexemes"]:
            assert lex in lexemes, (
                f"[{cid}] '{lex}' not found in Integer Constants Table"
            )

    # -- Symbol table: floats --
    if "float_lexemes" in case:
        entries = parse_table_entries(stdout, "Float Constants Table")
        lexemes = [e[1] for e in entries]
        for lex in case["float_lexemes"]:
            assert lex in lexemes, (
                f"[{cid}] '{lex}' not found in Float Constants Table"
            )

    # -- Symbol table: strings --
    if "string_lexemes" in case:
        entries = parse_table_entries(stdout, "String Constants Table")
        lexemes = [e[1] for e in entries]
        for lex in case["string_lexemes"]:
            assert lex in lexemes, (
                f"[{cid}] '{lex}' not found in String Constants Table"
            )


# ===========================================================================
# 1. Data-driven tests from test_cases.jsonl
# ===========================================================================
@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=[f"{c['category']}/{c['id']}" for c in TEST_CASES],
)
def test_case(case):
    """Generic data-driven test: run scanner, check assertions from JSONL."""
    check_case(case)


# ===========================================================================
# 2. Full curated dataset — every valid kernel scans error-free
# ===========================================================================
@pytest.mark.parametrize(
    "kernel", CURATED,
    ids=[k["function_name"] for k in CURATED],
)
def test_curated_kernel(kernel):
    """Valid curated kernel scans without lexer errors."""
    stdout, stderr = run_scanner(kernel["kernel_code"])
    tokens = parse_tokens(stdout)
    assert len(tokens) > 0, "Scanner produced no tokens"
    assert stderr.strip() == "", f"Unexpected errors:\n{stderr}"


# ===========================================================================
# 3. Adversarial — scanner-detectable errors
# ===========================================================================
@pytest.mark.parametrize(
    "kernel", ADVERSARIAL_SCANNER,
    ids=[f"{k['id']}[{k.get('subcategory','')}]" for k in ADVERSARIAL_SCANNER],
)
def test_adversarial_scanner_error(kernel):
    """Inputs with invalid characters must produce scanner errors."""
    _, stderr = run_scanner(kernel["kernel_code"])
    assert stderr.strip() != "", (
        f"Expected error for {kernel['id']}: {kernel.get('error_description','')}"
    )


# ===========================================================================
# 4. Adversarial — parser-level errors tokenize cleanly
# ===========================================================================
@pytest.mark.parametrize(
    "kernel", ADVERSARIAL_PARSER,
    ids=[f"{k['id']}[{k.get('subcategory','')}]" for k in ADVERSARIAL_PARSER],
)
def test_adversarial_parser_tokenizes(kernel):
    """Parser-level errors produce valid tokens without scanner errors."""
    stdout, _ = run_scanner(kernel["kernel_code"])
    tokens = parse_tokens(stdout)
    assert len(tokens) > 0, "Scanner produced no tokens"


# ===========================================================================
# 5. Summary reports
# ===========================================================================
class TestSummary:

    def test_curated_summary(self):
        passed, failed = 0, []
        for k in CURATED:
            stdout, stderr = run_scanner(k["kernel_code"])
            if parse_tokens(stdout) and stderr.strip() == "":
                passed += 1
            else:
                failed.append(k["function_name"])
        total = len(CURATED)
        print(f"\n{'='*50}")
        print(f"  CURATED: {passed}/{total} passed")
        print(f"{'='*50}")
        if failed:
            for name in failed[:10]:
                print(f"    FAIL: {name}")
        assert passed == total

    def test_adversarial_lexer_summary(self):
        detected, missed = 0, []
        for k in ADVERSARIAL_SCANNER:
            _, stderr = run_scanner(k["kernel_code"])
            if stderr.strip():
                detected += 1
            else:
                missed.append(k["id"])
        total = len(ADVERSARIAL_SCANNER)
        print(f"\n{'='*50}")
        print(f"  ADVERSARIAL (scanner): {detected}/{total} detected")
        print(f"{'='*50}")
        assert detected == total
