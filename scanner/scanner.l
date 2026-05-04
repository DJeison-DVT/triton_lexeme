/* ================================================================
 * scanner.l — Lexical Analyzer for Triton GPU Kernel Language
 *
 * TC3002B  Computer Science Advanced Applications Development
 * Module 3: Compilers — Lexical Analysis Phase (Step 7)
 *
 * This scanner recognizes all 47 terminal patterns defined in the
 * lexical specification of the Triton GPU kernel subset language.
 * It produces a numbered token stream and four symbol tables.
 *
 * Build:  flex scanner.l && gcc lex.yy.c -o scanner
 * Usage:  ./scanner <source_file>
 * ================================================================ */


/* ================================================================
 * 1. DEFINITION SECTION
 *
 * Contains C declarations (inside %{ ... %}) that are copied
 * verbatim into the generated scanner source file.  Here we
 * define token IDs, symbol-table data structures, global state,
 * and function prototypes.
 * ================================================================ */
%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------
 * Token IDs
 *
 * Each token type receives a unique integer constant.
 *
 *   1–7    Literals and structural tokens
 *   8–17   Keywords
 *   18     Compound token  (@triton.jit)
 *   19–24  Multi-character operators
 *   25–42  Single-character operators and punctuation
 * ---------------------------------------------------------------- */
#define TOK_FLOAT_NUM       1
#define TOK_NUMBER          2
#define TOK_STRING          3
#define TOK_IDENTIFIER      4
#define TOK_NEWLINE         5
#define TOK_INDENT1         6
#define TOK_INDENT2         7

#define TOK_KW_DEF          8
#define TOK_KW_FOR          9
#define TOK_KW_IN          10
#define TOK_KW_RANGE       11
#define TOK_KW_IF          12
#define TOK_KW_ELSE        13
#define TOK_KW_IS          14
#define TOK_KW_TRUE        15
#define TOK_KW_FALSE       16
#define TOK_KW_NONE        17

#define TOK_JIT_DECORATOR  18

#define TOK_OP_FLOOR_DIV   19
#define TOK_OP_LE          20
#define TOK_OP_GE          21
#define TOK_OP_EQ          22
#define TOK_OP_NEQ         23
#define TOK_OP_POW         24

#define TOK_LPAREN         25
#define TOK_RPAREN         26
#define TOK_LBRACKET       27
#define TOK_RBRACKET       28
#define TOK_COLON          29
#define TOK_COMMA          30
#define TOK_ASSIGN         31
#define TOK_OP_PLUS        32
#define TOK_OP_MINUS       33
#define TOK_OP_STAR        34
#define TOK_OP_DIV         35
#define TOK_OP_MOD         36
#define TOK_OP_AND         37
#define TOK_OP_XOR         38
#define TOK_OP_LT          39
#define TOK_OP_GT          40
#define TOK_DOT            41
#define TOK_AT             42

/* ----------------------------------------------------------------
 * Symbol Table
 *
 * Four separate tables store lexemes that carry semantic value:
 *   - Identifier table      (variable and function names)
 *   - Integer constants      (numeric integer literals)
 *   - Float constants        (numeric floating-point literals)
 *   - String constants       (quoted string literals)
 *
 * Each table uses insert-or-lookup semantics: when a lexeme is
 * encountered for the first time it receives a new row; subsequent
 * occurrences reuse the existing row.  The id field is a 1-based
 * sequential counter local to each table.
 * ---------------------------------------------------------------- */
#define MAX_TABLE_SIZE  1000
#define MAX_LEXEME_LEN  256

typedef struct {
    int   id;                        /* 1-based entry number       */
    char  lexeme[MAX_LEXEME_LEN];    /* the actual text            */
    int   first_line;                /* source line of first use   */
} SymEntry;

SymEntry id_table[MAX_TABLE_SIZE];      int id_count    = 0;
SymEntry int_table[MAX_TABLE_SIZE];     int int_count   = 0;
SymEntry float_table[MAX_TABLE_SIZE];   int float_count = 0;
SymEntry str_table[MAX_TABLE_SIZE];     int str_count   = 0;

/* ----------------------------------------------------------------
 * Scanner state
 * ---------------------------------------------------------------- */
int line_num    = 1;   /* current source line                      */
int paren_depth = 0;   /* nesting depth of ( ) and [ ]             */
int token_count = 0;   /* total tokens emitted so far              */

/* ----------------------------------------------------------------
 * Function prototypes  (defined in the User Code section)
 * ---------------------------------------------------------------- */
const char *token_name(int tok_id);
void  emit_token(int tok_id, const char *lexeme, int line);
int   sym_lookup(SymEntry table[], int count, const char *lexeme);
int   sym_insert(SymEntry table[], int *count, const char *lexeme, int line);
void  print_symbol_tables(void);
%}

/* Flex options — suppress warnings and remove need for -lfl */
%option noyywrap
%option nounput
%option noinput


%%
 /* ================================================================
  * 2. RULES SECTION
  *
  * Each rule is a pair   pattern { action }   where the pattern is
  * a regular expression and the action is C code executed when the
  * pattern matches.
  *
  * Ordering rationale (flex uses longest-match, first-listed wins
  * on ties):
  *   - Comments first            (discard before anything else)
  *   - Compound token            (longest literal, wins over @)
  *   - Indentation at line start (structural tokens)
  *   - Newlines                  (line counting)
  *   - Keywords before IDENTIFIER (same-length tie-break)
  *   - Float before integer      (longest match: 3.14 > 3)
  *   - Triple-quote strings before single-quote strings
  *   - IDENTIFIER
  *   - Multi-char operators before single-char operators
  *   - Single-char operators and punctuation
  *   - Whitespace                (discard)
  *   - Error catch-all
  * ================================================================ */

 /* ---- Comments: # to end of line, discarded (Section 3) -------- */
"#"[^\n]*               { /* discard comment */ }

 /* ---- Compound token: @triton.jit decorator -------------------- */
"@triton.jit"           { emit_token(TOK_JIT_DECORATOR, yytext, line_num); }

 /* ---- Indentation at line start --------------------------------
  * ^"        " (8 spaces) is listed before ^"    " (4 spaces) so
  * that flex's longest-match resolves eight-space lines to INDENT2
  * rather than two consecutive INDENT1 tokens.
  *
  * Inside unclosed parentheses or brackets (paren_depth > 0) the
  * indentation is decorative (continuation line) and is silently
  * consumed without emitting an INDENT token.
  * --------------------------------------------------------------- */
^"        "             { if (paren_depth == 0) emit_token(TOK_INDENT2, yytext, line_num); }
^"    "                 { if (paren_depth == 0) emit_token(TOK_INDENT1, yytext, line_num); }

 /* ---- Newlines: emit token, then advance line counter ---------- */
\n                      { emit_token(TOK_NEWLINE, "\\n", line_num); line_num++; }

 /* ---- Keywords -------------------------------------------------
  * Listed before the general identifier rule.  Flex picks the
  * first rule when two patterns match the same length, so "def"
  * wins over [a-zA-Z_][a-zA-Z0-9_]*.  For longer inputs such as
  * "define", the identifier rule wins via longest match.
  * --------------------------------------------------------------- */
"def"                   { emit_token(TOK_KW_DEF,   yytext, line_num); }
"for"                   { emit_token(TOK_KW_FOR,   yytext, line_num); }
"in"                    { emit_token(TOK_KW_IN,    yytext, line_num); }
"range"                 { emit_token(TOK_KW_RANGE, yytext, line_num); }
"if"                    { emit_token(TOK_KW_IF,    yytext, line_num); }
"else"                  { emit_token(TOK_KW_ELSE,  yytext, line_num); }
"is"                    { emit_token(TOK_KW_IS,    yytext, line_num); }
"True"                  { emit_token(TOK_KW_TRUE,  yytext, line_num); }
"False"                 { emit_token(TOK_KW_FALSE, yytext, line_num); }
"None"                  { emit_token(TOK_KW_NONE,  yytext, line_num); }

 /* ---- Literals ------------------------------------------------- */

 /* Float: decimal or scientific notation.
  * Pattern 1: [0-9]+.[0-9]+ with optional exponent  (e.g. 3.14, 1.5e-3)
  * Pattern 2: [0-9]+e[+-]?[0-9]+  integer with exponent  (e.g. 1e5)
  * Both listed before the integer rule so longest match resolves
  * 3.14 as FLOAT_NUM rather than NUMBER(3) DOT NUMBER(14).
  */
[0-9]+"."[0-9]+([eE]"-"?[0-9]+)?   { sym_insert(float_table, &float_count, yytext, line_num);
                                      emit_token(TOK_FLOAT_NUM, yytext, line_num); }

[0-9]+[eE]"-"?[0-9]+               { sym_insert(float_table, &float_count, yytext, line_num);
                                      emit_token(TOK_FLOAT_NUM, yytext, line_num); }

 /* Integer: [0-9]+ */
[0-9]+                  { sym_insert(int_table, &int_count, yytext, line_num);
                          emit_token(TOK_NUMBER, yytext, line_num); }

 /* Strings: triple-quoted variants listed first so that ''' wins
  * over ' on longest-match.  No escape sequences are processed.
  *
  * Triple-quoted strings may span multiple lines; the action
  * counts embedded newlines to keep line_num accurate.
  */
"'''"[^']*"'''"         { int start = line_num;
                          { int i; for (i = 0; i < yyleng; i++) if (yytext[i] == '\n') line_num++; }
                          sym_insert(str_table, &str_count, yytext, start);
                          emit_token(TOK_STRING, yytext, start); }

\"\"\"[^"]*\"\"\"       { int start = line_num;
                          { int i; for (i = 0; i < yyleng; i++) if (yytext[i] == '\n') line_num++; }
                          sym_insert(str_table, &str_count, yytext, start);
                          emit_token(TOK_STRING, yytext, start); }

"'"[^']*"'"             { sym_insert(str_table, &str_count, yytext, line_num);
                          emit_token(TOK_STRING, yytext, line_num); }

\"[^"]*\"               { sym_insert(str_table, &str_count, yytext, line_num);
                          emit_token(TOK_STRING, yytext, line_num); }

 /* ---- Identifier: [a-zA-Z_][a-zA-Z0-9_]* ----------------------
  * Listed after all keyword rules so that reserved words win the
  * same-length tie-break.  Module prefixes (tl, libdevice, etc.)
  * and attribute names pass through this rule; the parser or
  * semantic analysis distinguishes them via the DOT context.
  * --------------------------------------------------------------- */
[a-zA-Z_][a-zA-Z0-9_]* { sym_insert(id_table, &id_count, yytext, line_num);
                          emit_token(TOK_IDENTIFIER, yytext, line_num); }

 /* ---- Multi-character operators --------------------------------
  * Listed before their single-character prefixes.  Flex's longest-
  * match guarantees //, <=, >=, ==, != and ** are consumed as one
  * two-character token rather than two separate single-character
  * tokens.
  * --------------------------------------------------------------- */
"//"                    { emit_token(TOK_OP_FLOOR_DIV, yytext, line_num); }
"<="                    { emit_token(TOK_OP_LE,        yytext, line_num); }
">="                    { emit_token(TOK_OP_GE,        yytext, line_num); }
"=="                    { emit_token(TOK_OP_EQ,        yytext, line_num); }
"!="                    { emit_token(TOK_OP_NEQ,       yytext, line_num); }
"**"                    { emit_token(TOK_OP_POW,       yytext, line_num); }

 /* ---- Single-character operators and punctuation ---------------
  * LPAREN and LBRACKET increment paren_depth; their closing
  * counterparts decrement it.  This depth is used by the
  * indentation rules above to distinguish structural indentation
  * from continuation-line whitespace.
  * --------------------------------------------------------------- */
"("                     { paren_depth++; emit_token(TOK_LPAREN,   yytext, line_num); }
")"                     { if (paren_depth > 0) paren_depth--; emit_token(TOK_RPAREN,   yytext, line_num); }
"["                     { paren_depth++; emit_token(TOK_LBRACKET, yytext, line_num); }
"]"                     { if (paren_depth > 0) paren_depth--; emit_token(TOK_RBRACKET, yytext, line_num); }
":"                     { emit_token(TOK_COLON,    yytext, line_num); }
","                     { emit_token(TOK_COMMA,    yytext, line_num); }
"="                     { emit_token(TOK_ASSIGN,   yytext, line_num); }
"+"                     { emit_token(TOK_OP_PLUS,  yytext, line_num); }
"-"                     { emit_token(TOK_OP_MINUS, yytext, line_num); }
"*"                     { emit_token(TOK_OP_STAR,  yytext, line_num); }
"/"                     { emit_token(TOK_OP_DIV,   yytext, line_num); }
"%"                     { emit_token(TOK_OP_MOD,   yytext, line_num); }
"&"                     { emit_token(TOK_OP_AND,   yytext, line_num); }
"^"                     { emit_token(TOK_OP_XOR,   yytext, line_num); }
"<"                     { emit_token(TOK_OP_LT,    yytext, line_num); }
">"                     { emit_token(TOK_OP_GT,    yytext, line_num); }
"."                     { emit_token(TOK_DOT,      yytext, line_num); }
"@"                     { emit_token(TOK_AT,       yytext, line_num); }

 /* ---- Whitespace: spaces and tabs, discarded (Section 3) ------- */
[ \t]+                  { /* discard inline whitespace */ }

 /* ---- Error: any character not matched above ------------------- */
.                       { fprintf(stderr, ">>> Error lexico: caracter invalido '%c' (ASCII %d) en linea %d\n",
                                  yytext[0], yytext[0], line_num); }


%%
/* ================================================================
 * 3. USER CODE SECTION
 *
 * Contains auxiliary C functions copied verbatim into the
 * generated source file.  These implement:
 *   - token_name()         — lookup table from token ID to name
 *   - emit_token()         — formatted output of one token
 *   - sym_lookup/insert()  — symbol-table management
 *   - print_symbol_tables() — end-of-scan report
 *   - main()               — program entry point
 * ================================================================ */

/* ----------------------------------------------------------------
 * token_name — Maps a numeric token ID to its human-readable name.
 *
 * Implemented as a static array indexed by the token ID constant,
 * giving O(1) access.  The array size matches the highest token ID.
 * ---------------------------------------------------------------- */
const char *token_name(int tok_id)
{
    static const char *names[] = {
        "UNKNOWN",         /*  0  (unused)           */
        "FLOAT_NUM",       /*  1                     */
        "NUMBER",          /*  2                     */
        "STRING",          /*  3                     */
        "IDENTIFIER",      /*  4                     */
        "NEWLINE",         /*  5                     */
        "INDENT1",         /*  6                     */
        "INDENT2",         /*  7                     */
        "KW_DEF",          /*  8                     */
        "KW_FOR",          /*  9                     */
        "KW_IN",           /* 10                     */
        "KW_RANGE",        /* 11                     */
        "KW_IF",           /* 12                     */
        "KW_ELSE",         /* 13                     */
        "KW_IS",           /* 14                     */
        "KW_TRUE",         /* 15                     */
        "KW_FALSE",        /* 16                     */
        "KW_NONE",         /* 17                     */
        "JIT_DECORATOR",   /* 18                     */
        "OP_FLOOR_DIV",    /* 19                     */
        "OP_LE",           /* 20                     */
        "OP_GE",           /* 21                     */
        "OP_EQ",           /* 22                     */
        "OP_NEQ",          /* 23                     */
        "OP_POW",          /* 24                     */
        "LPAREN",          /* 25                     */
        "RPAREN",          /* 26                     */
        "LBRACKET",        /* 27                     */
        "RBRACKET",        /* 28                     */
        "COLON",           /* 29                     */
        "COMMA",           /* 30                     */
        "ASSIGN",          /* 31                     */
        "OP_PLUS",         /* 32                     */
        "OP_MINUS",        /* 33                     */
        "OP_STAR",         /* 34                     */
        "OP_DIV",          /* 35                     */
        "OP_MOD",          /* 36                     */
        "OP_AND",          /* 37                     */
        "OP_XOR",          /* 38                     */
        "OP_LT",           /* 39                     */
        "OP_GT",           /* 40                     */
        "DOT",             /* 41                     */
        "AT"               /* 42                     */
    };
    if (tok_id < 0 || tok_id > 42) return "UNKNOWN";
    return names[tok_id];
}

/* ----------------------------------------------------------------
 * emit_token — Prints one token record to stdout.
 *
 * Output format:
 *     No.  <TokenName, "lexeme", line>
 *
 * The sequential number (token_count) lets the reader trace the
 * exact position of each token in the stream.
 * ---------------------------------------------------------------- */
void emit_token(int tok_id, const char *lexeme, int line)
{
    token_count++;
    printf("%-5d <%-15s, \"%s\", %d>\n",
           token_count, token_name(tok_id), lexeme, line);
}

/* ----------------------------------------------------------------
 * sym_lookup — Linear search for a lexeme in a symbol table.
 *
 * Returns the 0-based index if found, -1 otherwise.
 * ---------------------------------------------------------------- */
int sym_lookup(SymEntry table[], int count, const char *lexeme)
{
    int i;
    for (i = 0; i < count; i++) {
        if (strcmp(table[i].lexeme, lexeme) == 0)
            return i;
    }
    return -1;
}

/* ----------------------------------------------------------------
 * sym_insert — Insert-or-lookup: adds a lexeme to the table only
 * if it is not already present.
 *
 * Returns the 0-based index of the (possibly existing) entry.
 * The id field is 1-based for readability in the printed tables.
 * ---------------------------------------------------------------- */
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

/* ----------------------------------------------------------------
 * print_symbol_tables — Prints all four tables after scanning.
 *
 * Tables are printed only if they contain at least one entry.
 * Columns match the design specification:
 *   ID | Lexeme | First Line
 * ---------------------------------------------------------------- */
void print_symbol_tables(void)
{
    int i;

    printf("\n========================================\n");
    printf("         SYMBOL TABLES\n");
    printf("========================================\n");

    /* ---- Identifier table ---- */
    printf("\n--- Identifier Table (%d entries) ---\n", id_count);
    printf("%-6s %-30s %-10s\n", "ID", "Lexeme", "First Line");
    printf("%-6s %-30s %-10s\n", "------", "------------------------------", "----------");
    for (i = 0; i < id_count; i++)
        printf("%-6d %-30s %-10d\n",
               id_table[i].id, id_table[i].lexeme, id_table[i].first_line);

    /* ---- Integer constants table ---- */
    printf("\n--- Integer Constants Table (%d entries) ---\n", int_count);
    printf("%-6s %-30s %-10s\n", "ID", "Lexeme", "First Line");
    printf("%-6s %-30s %-10s\n", "------", "------------------------------", "----------");
    for (i = 0; i < int_count; i++)
        printf("%-6d %-30s %-10d\n",
               int_table[i].id, int_table[i].lexeme, int_table[i].first_line);

    /* ---- Float constants table ---- */
    printf("\n--- Float Constants Table (%d entries) ---\n", float_count);
    printf("%-6s %-30s %-10s\n", "ID", "Lexeme", "First Line");
    printf("%-6s %-30s %-10s\n", "------", "------------------------------", "----------");
    for (i = 0; i < float_count; i++)
        printf("%-6d %-30s %-10d\n",
               float_table[i].id, float_table[i].lexeme, float_table[i].first_line);

    /* ---- String constants table ---- */
    printf("\n--- String Constants Table (%d entries) ---\n", str_count);
    printf("%-6s %-30s %-10s\n", "ID", "Lexeme", "First Line");
    printf("%-6s %-30s %-10s\n", "------", "------------------------------", "----------");
    for (i = 0; i < str_count; i++)
        printf("%-6d %-30s %-10d\n",
               str_table[i].id, str_table[i].lexeme, str_table[i].first_line);
}

/* ----------------------------------------------------------------
 * main — Entry point.
 *
 * Opens the source file given as a command-line argument (or reads
 * from stdin), runs the scanner via yylex(), and prints the four
 * symbol tables once scanning is complete.
 * ---------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <archivo_fuente>\n", argv[0]);
        return 1;
    }

    yyin = fopen(argv[1], "r");
    if (!yyin) {
        fprintf(stderr, "Error: no se puede abrir '%s'\n", argv[1]);
        return 1;
    }

    printf("========================================\n");
    printf("    TRITON LEXICAL ANALYZER OUTPUT\n");
    printf("========================================\n");
    printf("Archivo: %s\n", argv[1]);
    printf("========================================\n\n");
    printf("%-5s %-17s %-s\n", "No.", "Token", "Details");
    printf("-------------------------------------------\n");

    yylex();

    print_symbol_tables();

    printf("\n========================================\n");
    printf("Total de tokens: %d\n", token_count);
    printf("========================================\n");

    fclose(yyin);
    return 0;
}
