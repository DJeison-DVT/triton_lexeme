## 4. Implementation

This section presents the complete lex specification for the Triton GPU kernel scanner and explains every code element across the three canonical sections of a lex file: the Definition Section, the Rules Section, and the User Code Section. The source file is `scanner/scanner.l`. It is compiled with the standard toolchain:

```
flex scanner.l          # produces lex.yy.c
gcc lex.yy.c -o triton_scanner   # no external libraries required
```

The `%option noyywrap` directive eliminates the dependency on the flex runtime library (`-lfl`), making the build self-contained.

---

### 4.1 Definition Section (lines 16–136)

The Definition Section is enclosed between `%{` and `%}`. Its contents are copied verbatim into the generated C source file and are therefore standard C code. This section establishes three categories of declarations: token identifiers, data structures, and scanner state.

#### 4.1.1 Standard Headers (lines 25–27)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
```

- `stdio.h` provides `printf` and `fprintf` for token output and error messages, and the `FILE` type for `yyin`.
- `stdlib.h` is included for general utility; although the current implementation does not call `malloc`, the header is retained for potential future extensions.
- `string.h` provides `strcmp` for symbol-table lookup and `strncpy` for safe lexeme copying.

#### 4.1.2 Token IDs (lines 29–85)

Each token type receives a unique integer constant via `#define`. The 42 constants are organized into five groups that mirror the analysis document:

| Group | Range | Count | Examples |
|-------|-------|-------|----------|
| Literals and structural | 1–7 | 7 | `TOK_FLOAT_NUM`, `TOK_NUMBER`, `TOK_STRING`, `TOK_IDENTIFIER`, `TOK_NEWLINE`, `TOK_INDENT1`, `TOK_INDENT2` |
| Keywords | 8–17 | 10 | `TOK_KW_DEF`, `TOK_KW_FOR`, `TOK_KW_IN`, `TOK_KW_RANGE`, `TOK_KW_IF`, `TOK_KW_ELSE`, `TOK_KW_IS`, `TOK_KW_TRUE`, `TOK_KW_FALSE`, `TOK_KW_NONE` |
| Compound token | 18 | 1 | `TOK_JIT_DECORATOR` |
| Multi-character operators | 19–24 | 6 | `TOK_OP_FLOOR_DIV`, `TOK_OP_LE`, `TOK_OP_GE`, `TOK_OP_EQ`, `TOK_OP_NEQ`, `TOK_OP_POW` |
| Single-character operators | 25–42 | 18 | `TOK_LPAREN` through `TOK_AT` |

Note that while the design specifies 47 automata, several automata share token IDs. The four string-variant automata (single-quoted, double-quoted, triple-single, triple-double) all produce `TOK_STRING = 3`. The comment and whitespace automata do not produce tokens at all — they are recognized and discarded. This yields 42 distinct token IDs.

#### 4.1.3 Symbol Table Structure (lines 87–113)

The symbol table is implemented as four parallel arrays of `SymEntry` structs:

```c
typedef struct {
    int   id;                        /* 1-based entry number       */
    char  lexeme[MAX_LEXEME_LEN];    /* the actual text            */
    int   first_line;                /* source line of first use   */
} SymEntry;
```

The four tables correspond to the four categories identified in the design:

| Table | Variable | Stores |
|-------|----------|--------|
| Identifier | `id_table` | Variable names, function names, module prefixes, attribute names |
| Integer constants | `int_table` | Integer literal lexemes |
| Float constants | `float_table` | Floating-point literal lexemes |
| String constants | `str_table` | Quoted string literal lexemes |

Each table has a maximum capacity of `MAX_TABLE_SIZE = 1000` entries with lexemes up to `MAX_LEXEME_LEN = 256` characters. All tables use insert-or-lookup semantics: a lexeme is stored only on its first occurrence, and subsequent encounters reuse the existing entry.

#### 4.1.4 Scanner State (lines 115–120)

Three global variables track the scanner's runtime state:

- `line_num` (initialized to 1): the current source line, incremented on each `\n`.
- `paren_depth` (initialized to 0): the nesting depth of `(` `)` and `[` `]`. This counter is the mechanism by which the scanner distinguishes structural indentation from continuation-line whitespace. When `paren_depth > 0`, leading spaces at line start are silently consumed.
- `token_count` (initialized to 0): a sequential counter incremented by `emit_token`, providing each token with a unique ordinal in the output.

#### 4.1.5 Function Prototypes (lines 122–129)

Five functions are declared here and defined in the User Code Section:

- `token_name(int tok_id)` — maps a token ID to its human-readable name.
- `emit_token(int tok_id, const char *lexeme, int line)` — prints one token record.
- `sym_lookup(SymEntry[], int, const char*)` — searches a table for a lexeme.
- `sym_insert(SymEntry[], int*, const char*, int)` — insert-or-lookup into a table.
- `print_symbol_tables(void)` — prints all four tables at end of scan.

#### 4.1.6 Flex Options (lines 132–135)

```
%option noyywrap
%option nounput
%option noinput
```

- `noyywrap` generates an internal `yywrap()` that returns 1, eliminating the need for the flex library.
- `nounput` and `noinput` suppress compiler warnings about unused `yyunput()` and `input()` functions, which are not called by this scanner.

---

### 4.2 Rules Section (lines 138–293)

The Rules Section is delimited by `%%` markers. Each rule consists of a regular-expression pattern followed by a C action enclosed in braces. Flex compiles all patterns into a single deterministic finite automaton and, at runtime, applies two resolution principles:

1. **Longest match**: among all patterns that match the current input, the one consuming the most characters wins.
2. **First listed**: when two patterns match the same number of characters, the one appearing first in the specification wins.

These two principles determine the ordering of every rule in this section.

#### 4.2.1 Comments (line 163)

```
"#"[^\n]*               { /* discard comment */ }
```

The pattern matches `#` followed by zero or more non-newline characters. The action is empty — the matched text is consumed and no token is emitted. This implements the design decision that comments are non-token constructs discarded by the lexer. The comment rule is listed first so that a line like `# def` is consumed as a comment rather than being partially matched by keyword rules.

#### 4.2.2 Compound Token (line 166)

```
"@triton.jit"           { emit_token(TOK_JIT_DECORATOR, yytext, line_num); }
```

The 11-character literal `@triton.jit` is recognized as a single `JIT_DECORATOR` token. Flex's longest-match rule ensures this pattern wins over the single-character `@` rule (1 character) whenever the full decorator string is present. If the input contains `@` followed by anything other than `triton.jit`, the single-character `@` rule matches instead, producing an `AT` token.

#### 4.2.3 Indentation (lines 177–178)

```
^"        "             { if (paren_depth == 0) emit_token(TOK_INDENT2, yytext, line_num); }
^"    "                 { if (paren_depth == 0) emit_token(TOK_INDENT1, yytext, line_num); }
```

The `^` anchor restricts these patterns to the beginning of a line. The 8-space rule is listed before the 4-space rule so that longest match resolves an 8-space prefix as `INDENT2` rather than two consecutive `INDENT1` tokens.

The `paren_depth` guard implements the continuation-line strategy from the design: when scanning inside an unclosed parenthesis or bracket (`paren_depth > 0`), leading whitespace is decorative and is silently consumed without emitting an indent token. When `paren_depth == 0`, the whitespace is structural and produces the appropriate indent token for the parser.

This is the principal design decision of the lexer. The grammar permits only two structural indentation levels (4 spaces for the function body, 8 spaces for the for-loop body), making a fixed-level scheme sufficient. A stack-based INDENT/DEDENT mechanism, as used in full Python lexers, is unnecessary for this subset.

#### 4.2.4 Newlines (line 181)

```
\n                      { emit_token(TOK_NEWLINE, "\\n", line_num); line_num++; }
```

Each newline character emits a `NEWLINE` token at the current line number, then increments `line_num` for subsequent tokens. The `NEWLINE` token is required by the grammar as a statement separator within `body` and `for_body` rules.

#### 4.2.5 Keywords (lines 189–198)

```
"def"                   { emit_token(TOK_KW_DEF,   yytext, line_num); }
"for"                   { emit_token(TOK_KW_FOR,   yytext, line_num); }
...
"None"                  { emit_token(TOK_KW_NONE,  yytext, line_num); }
```

Ten keyword patterns are listed before the general identifier rule. Flex's first-listed tie-break ensures that when the input contains `def` followed by a non-identifier character (space, parenthesis, etc.), the keyword rule wins over the identifier rule, since both match exactly 3 characters.

For longer inputs such as `define`, the identifier pattern `[a-zA-Z_][a-zA-Z0-9_]*` matches 6 characters while `"def"` matches only 3. Longest match resolves this to `IDENTIFIER`, which is the correct behavior.

The words `import`, `from`, and `as` are deliberately absent from the keyword list. At the lexical level they are indistinguishable from user-defined identifiers; the parser distinguishes them by context within the import rules.

#### 4.2.6 Float Literals (lines 208–212)

```
[0-9]+"."[0-9]+([eE]"-"?[0-9]+)?   { ... emit_token(TOK_FLOAT_NUM, ...); }
[0-9]+[eE]"-"?[0-9]+               { ... emit_token(TOK_FLOAT_NUM, ...); }
```

Two patterns cover the float formats defined in the grammar:

1. Decimal notation with optional exponent: `3.14`, `1.5e-3`, `2.0E10`
2. Integer with mandatory exponent: `1e5`, `3e-2`

Both patterns are listed before the integer rule. For input `3.14`, the first float pattern matches 4 characters while the integer pattern matches only 1 (`3`). Longest match resolves this correctly to `FLOAT_NUM`, preventing the false tokenization `NUMBER(3) DOT(.) NUMBER(14)`.

Both actions call `sym_insert` to record the lexeme in the float constants table.

#### 4.2.7 Integer Literals (lines 215–216)

```
[0-9]+                  { sym_insert(int_table, &int_count, yytext, line_num);
                          emit_token(TOK_NUMBER, yytext, line_num); }
```

Matches one or more digits. This rule only fires when no float pattern produces a longer match. The action inserts the lexeme into the integer constants table.

#### 4.2.8 String Literals (lines 224–238)

```
"'''"[^']*"'''"         { ... emit_token(TOK_STRING, ...); }
\"\"\"[^"]*\"\"\"       { ... emit_token(TOK_STRING, ...); }
"'"[^']*"'"             { ... emit_token(TOK_STRING, ...); }
\"[^"]*\"               { ... emit_token(TOK_STRING, ...); }
```

Four patterns cover the string variants from the grammar, ordered from longest to shortest potential match:

1. Triple single-quoted: `'''...'''`
2. Triple double-quoted: `"""..."""`
3. Single-quoted: `'...'`
4. Double-quoted: `"..."`

Triple-quoted patterns are listed first so that `'''hello'''` matches the triple rule (11 characters) rather than the single rule matching `''` (2 characters, an empty string). This is a direct application of flex's longest-match principle.

The triple-quoted string actions include a loop to count embedded newline characters, keeping `line_num` accurate for multi-line strings:

```c
int start = line_num;
{ int i; for (i = 0; i < yyleng; i++) if (yytext[i] == '\n') line_num++; }
```

All four variants produce `TOK_STRING = 3` and insert the lexeme into the string constants table.

#### 4.2.9 Identifiers (lines 246–247)

```
[a-zA-Z_][a-zA-Z0-9_]* { sym_insert(id_table, &id_count, yytext, line_num);
                          emit_token(TOK_IDENTIFIER, yytext, line_num); }
```

This rule is deliberately listed after all keyword rules. For input `def`, both this rule and `"def"` match 3 characters; the keyword rule wins because it appears first. For input `define`, this rule matches 6 characters and wins by longest match.

Module prefixes (`tl`, `libdevice`, `tl_math`, `triton_helpers`) and attribute names (`load`, `store`, `dtype`) are all recognized as plain identifiers. The parser distinguishes them from user-defined names by the surrounding `DOT` context, as specified in the design.

#### 4.2.10 Multi-Character Operators (lines 255–260)

```
"//"                    { emit_token(TOK_OP_FLOOR_DIV, yytext, line_num); }
"<="                    { emit_token(TOK_OP_LE,        yytext, line_num); }
">="                    { emit_token(TOK_OP_GE,        yytext, line_num); }
"=="                    { emit_token(TOK_OP_EQ,        yytext, line_num); }
"!="                    { emit_token(TOK_OP_NEQ,       yytext, line_num); }
"**"                    { emit_token(TOK_OP_POW,       yytext, line_num); }
```

Six two-character operator patterns. Flex's longest-match ensures `==` (2 characters) wins over `=` (1 character), `//` wins over `/`, and `**` wins over `*`. Listing these before the single-character rules is a defensive convention that makes the priority self-documenting, even though flex's longest-match would resolve the ambiguity regardless of order.

#### 4.2.11 Single-Character Operators and Punctuation (lines 268–285)

```
"("                     { paren_depth++; emit_token(TOK_LPAREN,   yytext, line_num); }
")"                     { if (paren_depth > 0) paren_depth--; emit_token(TOK_RPAREN,   yytext, line_num); }
"["                     { paren_depth++; emit_token(TOK_LBRACKET, yytext, line_num); }
"]"                     { if (paren_depth > 0) paren_depth--; emit_token(TOK_RBRACKET, yytext, line_num); }
":"                     { emit_token(TOK_COLON,    yytext, line_num); }
...
"@"                     { emit_token(TOK_AT,       yytext, line_num); }
```

Eighteen single-character patterns. The opening delimiters `(` and `[` increment `paren_depth`; the closing delimiters `)` and `]` decrement it (with a floor of zero to handle malformed input gracefully). This depth counter drives the continuation-line logic in the indentation rules (Section 4.2.3).

#### 4.2.12 Whitespace (line 288)

```
[ \t]+                  { /* discard inline whitespace */ }
```

One or more spaces or tabs are consumed without emitting a token. This rule only fires for inline whitespace — leading whitespace at line start is handled by the indentation rules.

#### 4.2.13 Error Catch-All (lines 291–292)

```
.                       { fprintf(stderr, ">>> Error lexico: caracter invalido '%c' (ASCII %d) en linea %d\n",
                                  yytext[0], yytext[0], line_num); }
```

The `.` pattern matches any single character not matched by previous rules (note: `.` does not match `\n` in flex). The action prints a diagnostic to stderr identifying the invalid character, its ASCII code, and the line number. Scanning continues after the error — the scanner does not abort, allowing it to report multiple errors in a single pass.

---

### 4.3 User Code Section (lines 295–509)

The User Code Section appears after the second `%%` delimiter. Its contents are copied verbatim into the generated source file and contain the auxiliary functions and the `main()` entry point.

#### 4.3.1 token_name (lines 314–363)

```c
const char *token_name(int tok_id)
{
    static const char *names[] = {
        "UNKNOWN",         /*  0  (unused)           */
        "FLOAT_NUM",       /*  1                     */
        ...
        "AT"               /* 42                     */
    };
    if (tok_id < 0 || tok_id > 42) return "UNKNOWN";
    return names[tok_id];
}
```

A static array of 43 string pointers indexed by token ID provides O(1) name resolution. The bounds check ensures robustness against invalid IDs.

#### 4.3.2 emit_token (lines 374–379)

```c
void emit_token(int tok_id, const char *lexeme, int line)
{
    token_count++;
    printf("%-5d <%-15s, \"%s\", %d>\n",
           token_count, token_name(tok_id), lexeme, line);
}
```

Prints one token record in the format `No. <TokenName, "lexeme", line>`. The sequential number (`token_count`) allows unambiguous reference to any position in the token stream.

#### 4.3.3 sym_lookup (lines 386–394)

```c
int sym_lookup(SymEntry table[], int count, const char *lexeme)
{
    int i;
    for (i = 0; i < count; i++) {
        if (strcmp(table[i].lexeme, lexeme) == 0)
            return i;
    }
    return -1;
}
```

Linear search through a symbol table, comparing lexemes with `strcmp`. Returns the 0-based index if found, -1 otherwise. Linear search is justified for this application: Triton kernels typically contain fewer than one hundred unique identifiers, making the O(n) cost negligible relative to I/O.

#### 4.3.4 sym_insert (lines 403–421)

```c
int sym_insert(SymEntry table[], int *count, const char *lexeme, int line)
{
    int idx = sym_lookup(table, *count, lexeme);
    if (idx >= 0)
        return idx;                           /* already present   */

    if (*count >= MAX_TABLE_SIZE) {
        fprintf(stderr, "Error: tabla de simbolos llena\n");
        return -1;
    }

    idx = *count;
    table[idx].id = idx + 1;                  /* 1-based           */
    strncpy(table[idx].lexeme, lexeme, MAX_LEXEME_LEN - 1);
    table[idx].lexeme[MAX_LEXEME_LEN - 1] = '\0';
    table[idx].first_line = line;
    (*count)++;
    return idx;
}
```

Implements insert-or-lookup semantics. The function first delegates to `sym_lookup`; if the lexeme already exists, it returns the existing index without modification. Otherwise, it allocates a new row at position `*count`, assigns a 1-based `id`, copies the lexeme using `strncpy` with explicit null-termination for safety, and records the `first_line`. The overflow check prevents array-bounds violations.

#### 4.3.5 print_symbol_tables (lines 430–469)

Prints all four symbol tables in columnar format with headers showing the entry count. Each table displays three columns: ID, Lexeme, and First Line. Empty tables are still printed with their header to confirm they were initialized.

#### 4.3.6 main (lines 478–509)

```c
int main(int argc, char *argv[])
{
    if (argc < 2) { ... return 1; }
    yyin = fopen(argv[1], "r");
    if (!yyin) { ... return 1; }
    ...
    yylex();
    print_symbol_tables();
    ...
    fclose(yyin);
    return 0;
}
```

The entry point performs four steps:

1. Validates the command-line argument and opens the source file, assigning it to `yyin` (the flex input stream).
2. Prints a header with the filename.
3. Calls `yylex()`, the flex-generated scanner function, which runs the DFA and executes rule actions until EOF.
4. After scanning completes, calls `print_symbol_tables()` to output the four tables, prints the total token count, and closes the file.

---

### 4.4 Traceability to Analysis and Design

Every artifact produced during the analysis and design phases maps directly to a concrete element of the implementation:

| Design Artifact | Implementation Element |
|----------------|----------------------|
| 47 automata (Groups 1–5) | 47 flex patterns across the Rules Section; string variants share `TOK_STRING`, comment and whitespace have empty actions |
| 42 unique token IDs | 42 `#define TOK_*` constants (lines 40–85) |
| 4 symbol tables | 4 `SymEntry` arrays with insert-or-lookup functions |
| Insert-or-lookup semantics | `sym_lookup()` + `sym_insert()` with duplicate detection |
| Fixed-level indentation (INDENT1/INDENT2) | `^`-anchored rules with `paren_depth` guard (lines 177–178) |
| Continuation-line whitespace suppression | `paren_depth` counter incremented/decremented by `( ) [ ]` rules |
| Comment discarding | `"#"[^\n]*` with empty action (line 163) |
| Keyword priority over identifiers | Keywords listed before identifier rule; flex first-listed tie-break |
| Multi-char operator priority | Multi-char rules listed before single-char; flex longest-match |
| Error message generation | Catch-all `.` rule printing to stderr (lines 291–292) |
