## 5. Verification and Validation

This section describes the test model developed to verify and validate the Triton lexical analyzer. The approach emphasizes reproducibility, comprehensive coverage of the lexical specification, and a strict separation between test data and test logic.

### 5.1 Test Infrastructure

The testing strategy follows a **data-driven** design. All test cases are stored in a single file, `test_cases.jsonl`, which currently contains 92 entries encoded as one JSON object per line. The test runner, `test_scanner.py`, is a generic pytest harness that reads and interprets the JSONL file at collection time. Consequently, adding or modifying test cases requires editing only the data file; no changes to Python code are necessary.

The framework leverages **pytest** with its `parametrize` mechanism so that each JSONL entry appears as an individually named test in the output, enabling precise per-case reporting and filtering.

**JSONL entry format.** Every entry contains the following fields:

- **Metadata:** `id`, `category`, `subcategory`, `description` -- used for identification, filtering, and human-readable reporting.
- **Input:** `input` -- the source code string to be scanned.
- **Assertion fields** (all optional; an assertion is checked only when its corresponding field is present):
  - `has_types` -- a list of token types that must appear in the output.
  - `not_types` -- a list of token types that must **not** appear in the output.
  - `exact_tokens` -- an exact sequence match of the form `[[type, lexeme, line], ...]`.
  - `has_errors` -- a boolean indicating whether stderr should be non-empty (`true`) or empty (`false`).
  - `min_tokens` -- the minimum number of tokens expected.
  - `identifier_lexemes`, `integer_lexemes`, `float_lexemes`, `string_lexemes` -- assertions on symbol table contents.
  - `identifier_count` -- the exact number of rows expected in the identifier table.

This schema allows a single test runner to express a wide variety of correctness properties without requiring specialized Python assertions for each test category.

### 5.2 Test Case Design

The test cases are organized into categories, each targeting a specific aspect of the lexical specification. The subsections below describe what is tested in each category and provide justification for the design choices.

#### 5.2.1 Token Recognition (47 tests)

These tests verify that every token type defined in the lexical specification is correctly recognized by the scanner. There is one test per token type: each keyword, each operator, each literal type, and each structural token receives a dedicated entry.

**Justification.** This category ensures that every automaton described in the design section is implemented and produces the expected output. A missing or misnamed rule in the flex specification would cause the corresponding test to fail immediately.

**Example:**

```json
{"id": "tok_kw_def", "input": "def\n", "has_types": ["KW_DEF"]}
```

#### 5.2.2 Disambiguation (18 tests)

These tests exercise the critical ambiguity points where flex's resolution rules -- longest match and rule priority -- determine correctness:

- **Keyword vs. identifier:** The input `"define"` must produce `IDENTIFIER`, not `KW_DEF`. This validates that longest-match takes precedence over keyword rules.
- **Multi-character vs. single-character operators:** The input `"=="` must produce `OP_EQ`, not `ASSIGN` followed by `ASSIGN`. This validates longest-match among operator rules.
- **Float vs. integer:** The input `"3.14"` must produce `FLOAT_NUM`, not `NUMBER` + `DOT` + `NUMBER`.
- **Decorator recognition:** The input `"@triton.jit"` must produce `JIT_DECORATOR`, not `AT` + `IDENTIFIER` + `DOT` + `IDENTIFIER`.

**Justification.** Disambiguation is where the theoretical properties of the scanner generator (longest match, first-listed priority) interact with the concrete rule ordering in the flex file. Errors in this category would indicate either incorrect rule ordering or insufficient pattern specificity, making these tests essential for validating the design.

#### 5.2.3 Discarded Constructs (3 tests)

These tests verify that constructs designated as non-token-producing in the lexical specification are handled correctly:

- Comments produce no tokens.
- Whitespace produces no tokens.
- An exact token sequence verification confirms that `"x + y\n"` yields only the expected tokens with no extraneous entries.

**Justification.** This category validates Section 3 of the lexical specification, which defines constructs that are recognized but discarded by the scanner. Failure here would indicate that the scanner is leaking internal state into its output.

#### 5.2.4 Indentation (4 tests)

These tests verify the scanner's indentation-tracking mechanism:

- `INDENT1` is emitted for lines with 4-space indentation (function body lines).
- `INDENT2` is emitted for lines with 8-space indentation (for-loop body lines).
- Inside open parentheses or brackets, **no** indent tokens are emitted (continuation lines).

**Justification.** Indentation handling is the principal design decision in the scanner. The continuation-line suppression mechanism, implemented via a `paren_depth` counter, is the most complex stateful behavior in the lexer. These tests confirm that the mechanism operates correctly in both the base case and the suppression case.

#### 5.2.5 Error Handling (6 tests)

These tests present invalid characters to the scanner: `$`, `~`, `` ` ``, `?`, `\`, and `$` in expression context. Each test verifies that the scanner produces an error message on stderr.

**Justification.** The project rubric requires that the scanner "recognizes invalid symbols and provides the corresponding error message." These tests directly validate that requirement by confirming that each invalid character triggers the error-reporting path.

#### 5.2.6 Symbol Tables (5 tests)

These tests verify the insert-or-lookup semantics of the symbol tables:

- **Identifier deduplication:** The input `"x = x + x"` produces exactly one entry for `x` in the identifier table.
- **First-line tracking:** The input `"a\nb\na\n"` records `first_line=1` for `a`, confirming that the first occurrence is preserved.
- **Integer, float, and string table population:** Dedicated tests verify that each literal type is stored in its corresponding table.

**Justification.** The symbol tables implement insert-or-lookup semantics as described in the design. These tests verify that duplicates are suppressed and that metadata (specifically, the first line of occurrence) is recorded correctly.

#### 5.2.7 Integration -- Curated Kernels (100 tests)

Every kernel from the `curated_100.jsonl` dataset is scanned, and the test asserts that no errors are produced. The kernels are loaded directly from the parent project's JSONL file and are not duplicated in the test suite.

**Justification.** Unit tests verify individual rules in isolation; integration tests verify that the rules compose correctly on realistic inputs. The curated kernels are drawn from the kernelbook dataset and represent genuine Triton GPU kernel code, providing confidence that the scanner handles real-world inputs without failure.

#### 5.2.8 Integration -- Adversarial (20 tests)

These tests are divided into two groups based on the compiler phase that should detect the error:

- **Scanner-detectable errors (4 tests):** Inputs containing invalid characters (`$`, `?`), an unclosed string literal, and an unclosed brace. Each must produce stderr errors.
- **Parser-level errors (16 tests):** Inputs containing mismatched brackets, `===`, `::`, and similar constructs that are syntactically invalid but lexically valid. Each must tokenize without scanner errors.

**Justification.** This category validates that the scanner detects errors at the correct compiler phase boundary. A scanner that rejects lexically valid (but syntactically invalid) inputs would be overstepping its responsibility; a scanner that silently accepts invalid characters would be underperforming. These tests confirm the boundary is correctly drawn.

### 5.3 Results

The final test run completes with **226 out of 226 tests passing** in approximately 2.2 seconds.

| Category | Tests | Passed | Description |
|---|---|---|---|
| Token Recognition | 47 | 47 | All 42 token types verified |
| Disambiguation | 18 | 18 | Longest-match and priority rules |
| Discarded | 3 | 3 | Comments and whitespace |
| Indentation | 4 | 4 | Structural vs. continuation |
| Error Handling | 6 | 6 | Invalid character detection |
| Symbol Tables | 5 | 5 | Insert-or-lookup, deduplication |
| Curated Kernels | 100 | 100 | Real valid Triton kernels |
| Adversarial (scanner) | 4 | 4 | Invalid character detection |
| Adversarial (parser) | 16 | 16 | Parser-level errors tokenize correctly |
| Summaries | 2 | 2 | Aggregate pass rates |
| JSONL cases | 21 | 21 | Integration examples in test\_cases.jsonl |
| **Total** | **226** | **226** | |

### 5.4 Reproducibility

All tests can be reproduced from a clean build with the following commands:

```
cd scanner/
make clean && make
python3 -m pytest test_scanner.py -v
```

To filter by category for targeted debugging or review:

```
pytest test_scanner.py -k "disambiguation" -v
pytest test_scanner.py -k "curated" -v -s
```

The `-k` flag selects tests whose identifiers match the given substring, and the `-s` flag disables output capture, which is useful for inspecting scanner output during integration tests.
