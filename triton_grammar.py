"""
Triton kernel xgrammar validation helpers.

Loads the EBNF grammar from triton.ebnf and provides compile/validate
utilities backed by xgrammar.
"""

import pathlib

import xgrammar as xg

EBNF_PATH = pathlib.Path(__file__).parent / "triton.ebnf"


def load_ebnf() -> str:
    """Read the EBNF grammar file and return its contents."""
    return EBNF_PATH.read_text(encoding="utf-8")


def build_matcher() -> tuple[xg.GrammarMatcher, xg.CompiledGrammar]:
    """Compile the EBNF grammar and return a fresh GrammarMatcher."""
    grammar = xg.Grammar.from_ebnf(load_ebnf())
    vocab = [bytes([i]) for i in range(256)]
    tokenizer_info = xg.TokenizerInfo(
        vocab, vocab_type=xg.VocabType.RAW, stop_token_ids=[0]
    )
    compiler = xg.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_grammar(grammar)
    matcher = xg.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return matcher, compiled


def validate_code(code: str) -> bool:
    """Return True if `code` is accepted by the Triton grammar."""
    matcher, compiled = build_matcher()
    accepted = matcher.accept_string(code)
    return accepted and matcher.is_completed()
