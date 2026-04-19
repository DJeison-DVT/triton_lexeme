# Triton Lexeme — Reductive Grammar for Triton Code Generation

A reductive EBNF grammar (lexeme) for constraining LLM-generated [Triton](https://triton-lang.org/) GPU kernel code. Uses [xgrammar](https://github.com/mlc-ai/xgrammar) to validate that generated output conforms to the Triton kernel subset of Python.

## Purpose

When using LLMs to generate Triton kernels, unconstrained output can produce syntactically invalid code. This project defines a **reductive grammar** — an EBNF specification of the valid Triton kernel language surface — that can be plugged into xgrammar-based constrained decoding pipelines to guarantee syntactic correctness at generation time.

## Project Structure

```
lexeme/
├── triton.ebnf              # EBNF grammar definition (the main artifact)
├── triton_grammar.py        # xgrammar compile/validate helpers
├── test_triton_grammar.py   # Red-green test suite
├── extract_ops.py           # Extract whitelisted ops from curated dataset
├── curated_100.jsonl        # 100 valid Triton kernels (must be accepted)
├── adversarial_100.jsonl    # 100 invalid Triton snippets (must be rejected)
├── AGENTS.md                # Agent instructions
└── README.md
```

## Setup

Requires Python 3.12+.

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

## Running Tests

```bash
pytest test_triton_grammar.py -v
```

The test suite uses **red-green TDD**:

| Suite | Count | Expectation |
|---|---|---|
| `test_curated_kernel_accepted` | 100 | Grammar must **accept** every valid kernel |
| `test_adversarial_kernel_rejected` | 100 | Grammar must **reject** every invalid snippet |
| `TestGrammarSummary` | 2 | Aggregate pass/reject rates (must be 100%) |

## How It Works

1. **`triton.ebnf`** defines the grammar in EBNF format (llama.cpp-compatible).
2. **`validate_code(code)`** in `triton_grammar.py` loads and compiles the EBNF via xgrammar, then checks whether a given string is accepted:
   - `xg.Grammar.from_ebnf()` — parse the EBNF
   - `xg.GrammarCompiler` — compile against a byte-level tokenizer
   - `xg.GrammarMatcher.accept_string()` — validate the input
3. Tests parametrize over both datasets, giving per-kernel pass/fail visibility.

## Datasets

- **`curated_100.jsonl`** — Real-world Triton kernels from [kernelbook](https://huggingface.co/datasets/kernelbook). Each entry contains `kernel_code`, `function_name`, `tl_ops_used`, complexity metadata, and provenance.
- **`adversarial_100.jsonl`** — Hand-crafted invalid snippets covering lexer errors, structural errors, expression errors, statement errors, and indentation errors. Each entry has `should_parse: false` and an `error_description`.

## Whitelisted Operations

Extract the full list of namespaced ops found in the curated dataset:

```bash
python extract_ops.py
```

Output (29 unique ops across 4 namespaces):

| Namespace | Count | Operations |
|---|---|---|
| `tl` | 19 | `arange`, `broadcast_to`, `constexpr`, `debug_barrier`, `device_assert`, `float32`, `float64`, `full`, `int1`, `int8`, `int32`, `int64`, `load`, `num_programs`, `program_id`, `sigmoid`, `store`, `sum`, `where` |
| `libdevice` | 5 | `isnan`, `log1p`, `rsqrt`, `signbit`, `sqrt` |
| `tl_math` | 3 | `abs`, `exp`, `log` |
| `triton_helpers` | 2 | `maximum`, `minimum` |

## Development Workflow

1. Run `pytest` — observe which curated kernels are still RED
2. Edit `triton.ebnf` to expand the grammar
3. Re-run `pytest` — confirm new GREEN tests without breaking adversarial rejection
4. Repeat until all 202 tests pass
