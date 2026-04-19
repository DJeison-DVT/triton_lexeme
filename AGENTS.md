# Triton Lexeme — Agent Instructions

## Project Overview

This project defines a reductive EBNF grammar for Triton GPU kernel code and validates it using [xgrammar](https://github.com/mlc-ai/xgrammar). The goal is constrained decoding: ensuring LLM-generated Triton kernels are syntactically valid.

## Environment Setup

### Requirements

- Python 3.12+
- `xgrammar` and `pytest`

### Setup (any platform)

```bash
# From the lexeme/ directory
python3 -m venv .venv

# Activate
# Linux/macOS/WSL:
source .venv/bin/activate
# Windows (cmd):
.venv\Scripts\activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python3 -c "import xgrammar; print('xgrammar OK')"
python3 -c "import pytest; print('pytest OK')"
```

## Project Structure

```
lexeme/
├── triton.ebnf              # EBNF grammar definition (the main artifact)
├── triton_grammar.py        # xgrammar compile/validate helpers
├── test_triton_grammar.py   # Red-green test suite (202 tests)
├── extract_ops.py           # Extract whitelisted ops from curated dataset
├── curated_100.jsonl         # 100 valid Triton kernels (must be accepted)
├── adversarial_100.jsonl     # 100 invalid Triton snippets (must be rejected)
├── conftest.py              # Warns if branch is behind origin/main
├── requirements.txt         # Python dependencies
├── AGENTS.md
└── README.md
```

## Running Tests

```bash
# Full suite
pytest test_triton_grammar.py -v

# Summary only (fast feedback)
pytest test_triton_grammar.py::TestGrammarSummary -v -s

# Only curated (valid kernels)
pytest test_triton_grammar.py -k "curated" -v

# Only adversarial (invalid kernels)
pytest test_triton_grammar.py -k "adversarial" -v

# Single test by kernel name
pytest test_triton_grammar.py -k "triton_poi_fused_sum_0" -v
```

## Development Workflow

This project uses **red-green TDD**:

1. `git pull origin main` — sync to the latest grammar
2. Run `pytest` — observe which curated kernels are RED (rejected by grammar)
3. Edit `triton.ebnf` to expand the grammar
4. Re-run `pytest` — confirm new GREEN tests without breaking adversarial rejection
5. Commit and push

### Staying in sync

`conftest.py` automatically fetches `origin/main` before each test run. If your branch is behind, you'll see:

```
*** Your branch is N commit(s) behind origin/main. ***
*** Run 'git pull origin main' before testing the grammar. ***
```

Always pull before editing `triton.ebnf` to avoid merge conflicts on the single grammar file.

### Key rule

**Never edit `triton_grammar.py` or `test_triton_grammar.py` to make tests pass.** Only edit `triton.ebnf`.

## Grammar File (`triton.ebnf`)

The EBNF follows the [llama.cpp grammar format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md). Quick reference:

```ebnf
rule_name ::= expression           # rule definition
"literal"                          # exact string match
[a-zA-Z]                           # character class
rule_a | rule_b                    # alternation
rule_a rule_b                      # concatenation
rule?                              # optional (0 or 1)
rule*                              # zero or more
rule+                              # one or more
(rule_a rule_b)                    # grouping
```

## Whitelisted Operations

Extract the full whitelist from the curated dataset:

```bash
python extract_ops.py
```

The grammar must recognize these 29 ops:

| Namespace | Count | Operations |
|---|---|---|
| `tl` | 19 | `arange`, `broadcast_to`, `constexpr`, `debug_barrier`, `device_assert`, `float32`, `float64`, `full`, `int1`, `int8`, `int32`, `int64`, `load`, `num_programs`, `program_id`, `sigmoid`, `store`, `sum`, `where` |
| `libdevice` | 5 | `isnan`, `log1p`, `rsqrt`, `signbit`, `sqrt` |
| `tl_math` | 3 | `abs`, `exp`, `log` |
| `triton_helpers` | 2 | `maximum`, `minimum` |

## Validation API

```python
from triton_grammar import validate_code

code = '''@triton.jit
def my_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    tl.store(out_ptr0 + xindex, tmp0, xmask)'''

print(validate_code(code))  # True if grammar accepts it
```

## Dataset Format

### `curated_100.jsonl` (valid kernels)
```json
{
  "kernel_code": "@triton.jit\ndef ...",
  "function_name": "triton_poi_fused_sum_0",
  "tl_ops_used": ["tl.load", "tl.store", ...],
  "complexity_score": 1
}
```

### `adversarial_100.jsonl` (invalid kernels)
```json
{
  "id": "invalid_001",
  "category": "lexer_errors",
  "subcategory": "invalid_token_dollar",
  "kernel_code": "@triton.jit\ndef kernel():\n    x = 5 $ 3",
  "error_description": "Invalid token '$' not in language",
  "should_parse": false
}
```

Adversarial categories: `lexer_errors`, `structure_errors`, `expression_errors`, `statement_errors`, `indentation_errors` (20 each).
