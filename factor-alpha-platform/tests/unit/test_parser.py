"""
Unit tests for the expression parser.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.operators.parser import (
    AlphaExpressionParser,
    Lexer,
    Parser,
    NumberNode,
    IdentNode,
    BinaryOpNode,
    UnaryOpNode,
    FunctionCallNode,
)


class TestLexer:
    def test_simple_expression(self):
        tokens = Lexer("-returns").tokenize()
        assert tokens[0].value == "NEG"
        assert tokens[1].value == "returns"

    def test_function_call(self):
        tokens = Lexer("rank(close)").tokenize()
        assert tokens[0].value == "rank"
        assert tokens[1].value == "("
        assert tokens[2].value == "close"
        assert tokens[3].value == ")"

    def test_nested_function(self):
        tokens = Lexer("rank(delta(close, 5))").tokenize()
        names = [t.value for t in tokens if t.type != "EOF"]
        assert "rank" in names
        assert "delta" in names
        assert "close" in names
        assert 5.0 in names

    def test_arithmetic(self):
        tokens = Lexer("close / volume + 1").tokenize()
        ops = [t.value for t in tokens if t.type == "OP"]
        assert "/" in ops
        assert "+" in ops


class TestParser:
    def test_number(self):
        tokens = Lexer("42").tokenize()
        ast = Parser(tokens).parse()
        assert isinstance(ast, NumberNode)
        assert ast.value == 42.0

    def test_negative_number(self):
        tokens = Lexer("-5.0").tokenize()
        ast = Parser(tokens).parse()
        assert isinstance(ast, NumberNode)
        assert ast.value == -5.0

    def test_binary_op(self):
        tokens = Lexer("close / volume").tokenize()
        ast = Parser(tokens).parse()
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "/"

    def test_function_call(self):
        tokens = Lexer("rank(close)").tokenize()
        ast = Parser(tokens).parse()
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "rank"
        assert len(ast.args) == 1

    def test_nesting(self):
        tokens = Lexer("rank(delta(close, 5))").tokenize()
        ast = Parser(tokens).parse()
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "rank"
        inner = ast.args[0]
        assert isinstance(inner, FunctionCallNode)
        assert inner.name == "delta"

    def test_unary_minus_expression(self):
        tokens = Lexer("-rank(close)").tokenize()
        ast = Parser(tokens).parse()
        assert isinstance(ast, UnaryOpNode)
        assert ast.op == "-"


class TestAlphaExpressionParser:
    """Integration tests for the full parse → compile → execute pipeline."""

    def test_negative_returns(self, tiny_ctx):
        """'-returns' should negate the returns cross-section."""
        parser = AlphaExpressionParser(universe="TOP3000", lookback=20)
        compute_fn = parser.parse("-returns")

        date = tiny_ctx._trading_days[50]
        result = compute_fn(date, tiny_ctx)

        assert isinstance(result, dict)
        assert len(result) > 0

        # Verify values are negated returns
        for ticker, val in list(result.items())[:5]:
            actual_return = tiny_ctx.get_price(ticker, "returns", date)
            if not np.isnan(actual_return) and not np.isnan(val):
                assert val == pytest.approx(-actual_return, abs=1e-10)

    def test_rank_close(self, tiny_ctx):
        """'rank(close)' should rank by closing prices."""
        parser = AlphaExpressionParser(universe="TOP3000", lookback=20)
        compute_fn = parser.parse("rank(close)")

        date = tiny_ctx._trading_days[50]
        result = compute_fn(date, tiny_ctx)

        assert isinstance(result, dict)
        # All values should be in [0, 1]
        for v in result.values():
            if not np.isnan(v):
                assert 0 <= v <= 1.0

    def test_arithmetic_expression(self, tiny_ctx):
        """'close / volume' should compute ratio."""
        parser = AlphaExpressionParser(universe="TOP3000", lookback=20)
        compute_fn = parser.parse("close / volume")

        date = tiny_ctx._trading_days[50]
        result = compute_fn(date, tiny_ctx)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_delta_operator(self, tiny_ctx):
        """'delta(close, 5)' should compute 5-day price change."""
        parser = AlphaExpressionParser(universe="TOP3000", lookback=20)
        compute_fn = parser.parse("delta(close, 5)")

        date = tiny_ctx._trading_days[50]
        result = compute_fn(date, tiny_ctx)

        assert isinstance(result, dict)
        assert len(result) > 0
