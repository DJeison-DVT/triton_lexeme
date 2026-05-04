## Scanner Output Examples (Step 8)

This section presents the complete output of the Triton lexical analyzer on three representative inputs. Each example demonstrates a different aspect of the scanner's behavior: basic tokenization and symbol table construction, advanced language features and operator disambiguation, and error recovery. All outputs were produced by the compiled scanner binary (`triton_scanner`) operating on the indicated source files.

### Example 1: Simple Kernel --- Vector Copy

The first example exercises the core tokenization path: the `@triton.jit` decorator, function definition with typed parameters, integer arithmetic, Triton API calls (`tl.program_id`, `tl.arange`, `tl.load`, `tl.store`), comparison, assignment, and slice notation. The input contains no imports, no floating-point literals, no strings, and no nested control flow.

**Input** (`test_input.triton`):

```python
@triton.jit
def triton_poi_fused_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)
```

**Complete token stream** (93 tokens):

```
1     <JIT_DECORATOR  , "@triton.jit", 1>
2     <NEWLINE        , "\n", 1>
3     <KW_DEF         , "def", 2>
4     <IDENTIFIER     , "triton_poi_fused_sum_0", 2>
5     <LPAREN         , "(", 2>
6     <IDENTIFIER     , "in_ptr0", 2>
7     <COMMA          , ",", 2>
8     <IDENTIFIER     , "out_ptr0", 2>
9     <COMMA          , ",", 2>
10    <IDENTIFIER     , "xnumel", 2>
11    <COMMA          , ",", 2>
12    <IDENTIFIER     , "XBLOCK", 2>
13    <COLON          , ":", 2>
14    <IDENTIFIER     , "tl", 2>
15    <DOT            , ".", 2>
16    <IDENTIFIER     , "constexpr", 2>
17    <RPAREN         , ")", 2>
18    <COLON          , ":", 2>
19    <NEWLINE         , "\n", 2>
20    <INDENT1        , "    ", 3>
21    <IDENTIFIER     , "xnumel", 3>
22    <ASSIGN         , "=", 3>
23    <NUMBER         , "256", 3>
24    <NEWLINE        , "\n", 3>
25    <INDENT1        , "    ", 4>
26    <IDENTIFIER     , "xoffset", 4>
27    <ASSIGN         , "=", 4>
28    <IDENTIFIER     , "tl", 4>
29    <DOT            , ".", 4>
30    <IDENTIFIER     , "program_id", 4>
31    <LPAREN         , "(", 4>
32    <NUMBER         , "0", 4>
33    <RPAREN         , ")", 4>
34    <OP_STAR        , "*", 4>
35    <IDENTIFIER     , "XBLOCK", 4>
36    <NEWLINE        , "\n", 4>
37    <INDENT1        , "    ", 5>
38    <IDENTIFIER     , "xindex", 5>
39    <ASSIGN         , "=", 5>
40    <IDENTIFIER     , "xoffset", 5>
41    <OP_PLUS        , "+", 5>
42    <IDENTIFIER     , "tl", 5>
43    <DOT            , ".", 5>
44    <IDENTIFIER     , "arange", 5>
45    <LPAREN         , "(", 5>
46    <NUMBER         , "0", 5>
47    <COMMA          , ",", 5>
48    <IDENTIFIER     , "XBLOCK", 5>
49    <RPAREN         , ")", 5>
50    <LBRACKET       , "[", 5>
51    <COLON          , ":", 5>
52    <RBRACKET       , "]", 5>
53    <NEWLINE        , "\n", 5>
54    <INDENT1        , "    ", 6>
55    <IDENTIFIER     , "xmask", 6>
56    <ASSIGN         , "=", 6>
57    <IDENTIFIER     , "xindex", 6>
58    <OP_LT          , "<", 6>
59    <IDENTIFIER     , "xnumel", 6>
60    <NEWLINE        , "\n", 6>
61    <INDENT1        , "    ", 7>
62    <IDENTIFIER     , "x0", 7>
63    <ASSIGN         , "=", 7>
64    <IDENTIFIER     , "xindex", 7>
65    <NEWLINE        , "\n", 7>
66    <INDENT1        , "    ", 8>
67    <IDENTIFIER     , "tmp0", 8>
68    <ASSIGN         , "=", 8>
69    <IDENTIFIER     , "tl", 8>
70    <DOT            , ".", 8>
71    <IDENTIFIER     , "load", 8>
72    <LPAREN         , "(", 8>
73    <IDENTIFIER     , "in_ptr0", 8>
74    <OP_PLUS        , "+", 8>
75    <IDENTIFIER     , "x0", 8>
76    <COMMA          , ",", 8>
77    <IDENTIFIER     , "xmask", 8>
78    <RPAREN         , ")", 8>
79    <NEWLINE        , "\n", 8>
80    <INDENT1        , "    ", 9>
81    <IDENTIFIER     , "tl", 9>
82    <DOT            , ".", 9>
83    <IDENTIFIER     , "store", 9>
84    <LPAREN         , "(", 9>
85    <IDENTIFIER     , "out_ptr0", 9>
86    <OP_PLUS        , "+", 9>
87    <IDENTIFIER     , "x0", 9>
88    <COMMA          , ",", 9>
89    <IDENTIFIER     , "tmp0", 9>
90    <COMMA          , ",", 9>
91    <IDENTIFIER     , "xmask", 9>
92    <RPAREN         , ")", 9>
93    <NEWLINE        , "\n", 9>
```

**Symbol tables:**

```
Identifier Table (16 entries):
ID     Lexeme                         First Line
------ ------------------------------ ----------
1      triton_poi_fused_sum_0         2
2      in_ptr0                        2
3      out_ptr0                       2
4      xnumel                         2
5      XBLOCK                         2
6      tl                             2
7      constexpr                      2
8      xoffset                        4
9      program_id                     4
10     xindex                         5
11     arange                         5
12     xmask                          6
13     x0                             7
14     tmp0                           8
15     load                           8
16     store                          9

Integer Constants Table (2 entries):
ID     Lexeme                         First Line
------ ------------------------------ ----------
1      256                            3
2      0                              4

Float Constants Table (0 entries):
ID     Lexeme                         First Line
------ ------------------------------ ----------

String Constants Table (0 entries):
ID     Lexeme                         First Line
------ ------------------------------ ----------

Total tokens: 93
```

**Annotations.**

1. **`@triton.jit` as a single compound token.** The decorator is recognized as one `JIT_DECORATOR` token (token 1), not as three separate tokens (`AT`, `IDENTIFIER`, `DOT`, `IDENTIFIER`). This is achieved by placing the literal pattern `"@triton.jit"` before the single-character `@` rule in the flex specification. Flex's longest-match semantics guarantee that the 11-character compound pattern always wins over the 1-character `@` prefix.

2. **`tl.constexpr` decomposed into IDENTIFIER DOT IDENTIFIER.** The type annotation `tl.constexpr` on line 2 is not a keyword or a compound token. The scanner produces three tokens: `IDENTIFIER("tl")`, `DOT(".")`, and `IDENTIFIER("constexpr")` (tokens 14--16). Semantic resolution of module-qualified names is deferred to the parser or later compiler phases.

3. **`INDENT1` at every body line.** Every line inside the function body (lines 3--9) begins with a 4-space indentation, and each one produces an `INDENT1` token (tokens 20, 25, 37, 54, 61, 66, 80). This structural token allows a parser to recognize block boundaries without a separate indentation-tracking pass.

4. **Insert-or-lookup deduplication.** The integer literal `0` appears three times in the source (lines 4, 5, and 5 as the argument to `tl.program_id` and `tl.arange`), yet the Integer Constants Table contains only a single entry with `First Line = 4`. Similarly, the identifier `tl` appears six times but occupies a single row in the Identifier Table. This insert-or-lookup behavior prevents redundant entries and records only the first occurrence of each lexeme.

---

### Example 2: Complex Kernel Highlights

The second example exercises the full breadth of the lexical specification: import statements, continuation lines inside open parentheses, string literals, floating-point literals (decimal and scientific notation), all six multi-character operators, a `for` loop with `range`, double indentation (`INDENT2`), method chaining, and the `None` keyword.

**Input** (`test_complex.triton`):

```python
import triton
import triton.language as tl
@triton.jit
def triton_poi_fused_complex(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 - tmp1
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp1
    tmp7 = libdevice.isnan(tmp6).to(tl.int1)
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp6)
    tmp10 = 1e-16
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 >= tmp1
    tmp13 = tmp0 <= tmp1
    tmp14 = tmp0 == tmp1
    tmp15 = tmp0 != tmp1
    tmp16 = tmp0 // tmp1
    tmp17 = tmp0 ** tmp1
    for i in range(0, xnumel, 1):
        tmp18 = tl.load(in_ptr0 + i, None)
        tl.store(out_ptr0 + i, tmp18, None)
    tl.store(out_ptr0 + x0, tmp9, xmask)
```

The scanner produces **292 tokens** in total. Rather than reproduce the entire stream, the following highlights illustrate the most instructive tokenization decisions.

#### Highlight 1: Import lines (tokens 1--10)

```
1     <IDENTIFIER     , "import", 1>
2     <IDENTIFIER     , "triton", 1>
3     <NEWLINE        , "\n", 1>
4     <IDENTIFIER     , "import", 2>
5     <IDENTIFIER     , "triton", 2>
6     <DOT            , ".", 2>
7     <IDENTIFIER     , "language", 2>
8     <IDENTIFIER     , "as", 2>
9     <IDENTIFIER     , "tl", 2>
10    <NEWLINE        , "\n", 2>
```

The words `import` and `as` are not keywords in the Triton grammar; they are recognized as ordinary `IDENTIFIER` tokens. This is correct: `import` is handled by the Python runtime environment, not by the Triton kernel compiler. The scanner's keyword set (`def`, `for`, `in`, `range`, `if`, `else`, `is`, `True`, `False`, `None`) is intentionally minimal, limited to the constructs that appear inside kernel function bodies. Note also that `import` (6 characters) is resolved by flex's longest-match rule: the keyword `in` (2 characters) could match the first two characters, but the longer identifier match prevails.

#### Highlight 2: Continuation line (tokens 25--32)

```
24    <NEWLINE        , "\n", 4>
25    <IDENTIFIER     , "XBLOCK", 5>
26    <COLON          , ":", 5>
27    <IDENTIFIER     , "tl", 5>
28    <DOT            , ".", 5>
29    <IDENTIFIER     , "constexpr", 5>
30    <RPAREN         , ")", 5>
31    <COLON          , ":", 5>
32    <NEWLINE        , "\n", 5>
```

Line 5 of the source is indented with 4 spaces, but no `INDENT1` token appears. The function definition's opening parenthesis on line 4 (token 15) incremented `paren_depth` to 1, suppressing indentation tokens on all continuation lines until the matching closing parenthesis (token 30) restores it to 0. This mechanism ensures that only structurally meaningful indentation produces tokens.

#### Highlight 3: String literal (token 108)

```
106   <IDENTIFIER     , "eviction_policy", 12>
107   <ASSIGN         , "=", 12>
108   <STRING         , "'evict_last'", 12>
109   <RPAREN         , ")", 12>
```

The single-quoted string `'evict_last'` is recognized as a single `STRING` token, including the enclosing quote characters. The lexeme is stored in the String Constants Table. This token demonstrates that the scanner correctly handles keyword arguments with string values, a pattern common in Triton API calls such as `tl.load` with an `eviction_policy` parameter.

#### Highlight 4: Float literals (tokens 114, 167, 186)

```
114   <FLOAT_NUM      , "1.0", 13>
...
167   <FLOAT_NUM      , "0.0", 19>
...
186   <FLOAT_NUM      , "1e-16", 21>
```

Three distinct floating-point formats are represented: standard decimal notation (`1.0`, `0.0`) and scientific notation without a decimal point (`1e-16`). The scanner's flex specification includes two float patterns: `[0-9]+"."[0-9]+([eE]"-"?[0-9]+)?` for decimal floats and `[0-9]+[eE]"-"?[0-9]+` for pure scientific notation. Both patterns are listed before the integer rule, so flex's longest-match semantics prevent `1.0` from being split into `NUMBER("1")`, `DOT(".")`, `NUMBER("0")`.

#### Highlight 5: Multi-character operators (tokens 199, 206, 213, 220, 227, 234)

```
199   <OP_GE          , ">=", 23>
...
206   <OP_LE          , "<=", 24>
...
213   <OP_EQ          , "==", 25>
...
220   <OP_NEQ         , "!=", 26>
...
227   <OP_FLOOR_DIV   , "//", 27>
...
234   <OP_POW         , "**", 28>
```

All six multi-character operators are present in lines 23--28, each exercising the longest-match disambiguation. For example, `>=` is recognized as a single `OP_GE` token rather than `OP_GT` followed by `ASSIGN`. The multi-character operator rules are placed before their single-character counterparts in the flex specification, but it is the longest-match property---not rule ordering---that resolves the ambiguity, since the two-character patterns always consume more input than the one-character alternatives.

#### Highlight 6: For loop (tokens 237--250)

```
237   <INDENT1        , "    ", 29>
238   <KW_FOR         , "for", 29>
239   <IDENTIFIER     , "i", 29>
240   <KW_IN          , "in", 29>
241   <KW_RANGE       , "range", 29>
242   <LPAREN         , "(", 29>
243   <NUMBER         , "0", 29>
244   <COMMA          , ",", 29>
245   <IDENTIFIER     , "xnumel", 29>
246   <COMMA          , ",", 29>
247   <NUMBER         , "1", 29>
248   <RPAREN         , ")", 29>
249   <COLON          , ":", 29>
250   <NEWLINE        , "\n", 29>
```

The `for` loop header produces four keyword tokens: `KW_FOR`, `KW_IN`, and `KW_RANGE`, along with the loop variable `i` as `IDENTIFIER`. The keyword `in` (2 characters) is correctly distinguished from `in_ptr0` (7 characters) by longest-match: on line 29, `in` is followed by a space, so the two-character match wins and the keyword rule takes priority via first-listed tie-breaking.

#### Highlight 7: `INDENT2` in for-loop body (tokens 251, 265)

```
251   <INDENT2        , "        ", 30>
252   <IDENTIFIER     , "tmp18", 30>
...
265   <INDENT2        , "        ", 31>
266   <IDENTIFIER     , "tl", 31>
```

Lines 30--31, which form the body of the `for` loop, are indented with 8 spaces. The scanner emits `INDENT2` tokens for these lines. The 8-space indentation rule is listed before the 4-space rule in the flex specification, so flex's longest-match resolves `"        "` (8 spaces) as a single `INDENT2` rather than two consecutive `INDENT1` tokens.

**Symbol tables:**

```
Identifier Table (48 entries):
ID     Lexeme                         First Line
------ ------------------------------ ----------
1      import                         1
2      triton                         1
3      language                       2
4      as                             2
5      tl                             2
6      triton_poi_fused_complex       4
7      in_ptr0                        4
8      in_ptr1                        4
9      out_ptr0                       4
10     xnumel                         4
11     XBLOCK                         5
12     constexpr                      5
13     xoffset                        7
14     program_id                     7
15     xindex                         8
16     arange                         8
17     xmask                          9
18     x0                             10
19     tmp0                           11
20     load                           11
21     tmp1                           12
22     eviction_policy                12
23     tmp2                           13
24     tmp3                           14
25     tmp4                           15
26     tl_math                        16
27     exp                            16
28     tmp5                           16
29     tmp6                           17
30     libdevice                      18
31     isnan                          18
32     tmp7                           18
33     to                             18
34     int1                           18
35     tmp8                           19
36     where                          20
37     tmp9                           20
38     tmp10                          21
39     tmp11                          22
40     tmp12                          23
41     tmp13                          24
42     tmp14                          25
43     tmp15                          26
44     tmp16                          27
45     tmp17                          28
46     i                              29
47     tmp18                          30
48     store                          31

Integer Constants Table (3 entries):
ID     Lexeme                         First Line
------ ------------------------------ ----------
1      1024                           6
2      0                              7
3      1                              29

Float Constants Table (3 entries):
ID     Lexeme                         First Line
------ ------------------------------ ----------
1      1.0                            13
2      0.0                            19
3      1e-16                          21

String Constants Table (1 entry):
ID     Lexeme                         First Line
------ ------------------------------ ----------
1      'evict_last'                   12

Total tokens: 292
```

---

### Example 3: Error Detection

The third example demonstrates the scanner's error recovery behavior when encountering an invalid character.

**Input** (`test_error.triton`):

```python
@triton.jit
def kernel():
    x = 5 $ 3
```

**stderr output:**

```
>>> Error lexico: caracter invalido '$' (ASCII 36) en linea 3
```

**stdout token stream** (14 tokens):

```
1     <JIT_DECORATOR  , "@triton.jit", 1>
2     <NEWLINE        , "\n", 1>
3     <KW_DEF         , "def", 2>
4     <IDENTIFIER     , "kernel", 2>
5     <LPAREN         , "(", 2>
6     <RPAREN         , ")", 2>
7     <COLON          , ":", 2>
8     <NEWLINE        , "\n", 2>
9     <INDENT1        , "    ", 3>
10    <IDENTIFIER     , "x", 3>
11    <ASSIGN         , "=", 3>
12    <NUMBER         , "5", 3>
13    <NUMBER         , "3", 3>
14    <NEWLINE        , "\n", 3>
```

**Annotations.**

1. **The scanner does not abort on error.** Upon encountering the `$` character, the scanner reports the error to stderr and advances past the invalid character. Scanning then resumes with the next character in the input stream. This design follows standard lexical-analysis practice: a single invalid character should not prevent the scanner from processing the remainder of the source file, as downstream compiler phases may benefit from the additional token context for error reporting.

2. **Tokens before and after the error are recognized correctly.** Token 12 (`NUMBER "5"`) immediately precedes the invalid character, and token 13 (`NUMBER "3"`) immediately follows it. Both are correctly identified, demonstrating that the error catch-all rule in the flex specification (the final `.` pattern) consumes exactly one character and does not disrupt the scanner's state machine.

3. **The error message includes diagnostic detail.** The message reports the invalid character (`$`), its ASCII code (36), and its line number (3). This information is sufficient for a developer to locate and correct the error in the source file. The error is written to stderr, keeping the token stream on stdout clean and parseable by downstream tools.
