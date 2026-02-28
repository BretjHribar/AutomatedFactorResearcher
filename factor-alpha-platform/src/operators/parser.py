"""
Alpha Expression Parser — converts WebSim-style expression strings into callable functions.

Supports:
- Arithmetic: +, -, *, /, ^
- Comparison: <, <=, >, >=, ==, !=
- Logical: ||, &&, !
- Ternary: cond ? expr1 : expr2
- All operators from the operators library
- All data fields (close, volume, returns, etc.)
- Nested expressions to arbitrary depth
- Negative numbers and unary minus

Usage:
    parser = AlphaExpressionParser()
    compute_fn = parser.parse("-rank(delta(close, 5))")
    # Returns a callable: (date, DataContext) -> dict[str, float]
"""

from __future__ import annotations

import ast
import datetime as dt
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.operators import cross_sectional as cs
from src.operators import time_series as ts
from src.operators import element_wise as ew


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType:
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    OP = "OP"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    QUESTION = "QUESTION"
    COLON = "COLON"
    EOF = "EOF"


class Token:
    def __init__(self, type_: str, value: Any):
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    """Tokenize a WebSim expression string."""

    OPERATORS = {"||", "&&", "!=", "==", "<=", ">=", "<", ">", "+", "-", "*", "/", "^", "!"}

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _peek(self) -> str | None:
        if self.pos < len(self.text):
            return self.text[self.pos]
        return None

    def _advance(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while self.pos < len(self.text):
            ch = self.text[self.pos]

            if ch.isspace():
                self.pos += 1
                continue

            if ch == '(':
                tokens.append(Token(TokenType.LPAREN, "("))
                self.pos += 1
            elif ch == ')':
                tokens.append(Token(TokenType.RPAREN, ")"))
                self.pos += 1
            elif ch == ',':
                tokens.append(Token(TokenType.COMMA, ","))
                self.pos += 1
            elif ch == '?':
                tokens.append(Token(TokenType.QUESTION, "?"))
                self.pos += 1
            elif ch == ':':
                tokens.append(Token(TokenType.COLON, ":"))
                self.pos += 1
            elif ch in '0123456789' or (ch == '.' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
                tokens.append(self._read_number())
            elif ch == '-' and self._is_unary_minus(tokens):
                # Unary minus: treat as part of number or as OP
                if self.pos + 1 < len(self.text) and (self.text[self.pos + 1].isdigit() or self.text[self.pos + 1] == '.'):
                    self.pos += 1
                    num_tok = self._read_number()
                    num_tok.value = -num_tok.value
                    tokens.append(num_tok)
                else:
                    tokens.append(Token(TokenType.OP, "NEG"))
                    self.pos += 1
            elif self._match_two_char_op():
                op = self.text[self.pos:self.pos + 2]
                tokens.append(Token(TokenType.OP, op))
                self.pos += 2
            elif ch in "+-*/^<>=!":
                tokens.append(Token(TokenType.OP, ch))
                self.pos += 1
            elif ch.isalpha() or ch == '_':
                tokens.append(self._read_ident())
            else:
                self.pos += 1  # skip unknown

        tokens.append(Token(TokenType.EOF, None))
        return tokens

    def _read_number(self) -> Token:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self.pos += 1
        return Token(TokenType.NUMBER, float(self.text[start:self.pos]))

    def _read_ident(self) -> Token:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        return Token(TokenType.IDENT, self.text[start:self.pos])

    def _match_two_char_op(self) -> bool:
        if self.pos + 1 < len(self.text):
            two = self.text[self.pos:self.pos + 2]
            return two in ("||", "&&", "!=", "==", "<=", ">=")
        return False

    def _is_unary_minus(self, tokens: list[Token]) -> bool:
        """Check if '-' is unary (not binary subtraction)."""
        if not tokens:
            return True
        last = tokens[-1]
        return last.type in (TokenType.OP, TokenType.LPAREN, TokenType.COMMA, TokenType.QUESTION, TokenType.COLON)


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

class ASTNode:
    pass

class NumberNode(ASTNode):
    def __init__(self, value: float):
        self.value = value

class IdentNode(ASTNode):
    def __init__(self, name: str):
        self.name = name

class UnaryOpNode(ASTNode):
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand

class BinaryOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

class FunctionCallNode(ASTNode):
    def __init__(self, name: str, args: list[ASTNode]):
        self.name = name
        self.args = args

class TernaryNode(ASTNode):
    def __init__(self, condition: ASTNode, true_expr: ASTNode, false_expr: ASTNode):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------

class Parser:
    """Parse tokens into an AST."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, type_: str) -> Token:
        tok = self._current()
        if tok.type != type_:
            raise SyntaxError(f"Expected {type_}, got {tok}")
        self.pos += 1
        return tok

    def parse(self) -> ASTNode:
        node = self._parse_ternary()
        return node

    def _parse_ternary(self) -> ASTNode:
        node = self._parse_or()
        if self._current().type == TokenType.QUESTION:
            self.pos += 1
            true_expr = self._parse_ternary()
            self._eat(TokenType.COLON)
            false_expr = self._parse_ternary()
            return TernaryNode(node, true_expr, false_expr)
        return node

    def _parse_or(self) -> ASTNode:
        node = self._parse_and()
        while self._current().type == TokenType.OP and self._current().value == "||":
            self.pos += 1
            right = self._parse_and()
            node = BinaryOpNode("||", node, right)
        return node

    def _parse_and(self) -> ASTNode:
        node = self._parse_comparison()
        while self._current().type == TokenType.OP and self._current().value == "&&":
            self.pos += 1
            right = self._parse_comparison()
            node = BinaryOpNode("&&", node, right)
        return node

    def _parse_comparison(self) -> ASTNode:
        node = self._parse_add()
        while self._current().type == TokenType.OP and self._current().value in ("<", "<=", ">", ">=", "==", "!="):
            op = self._current().value
            self.pos += 1
            right = self._parse_add()
            node = BinaryOpNode(op, node, right)
        return node

    def _parse_add(self) -> ASTNode:
        node = self._parse_mul()
        while self._current().type == TokenType.OP and self._current().value in ("+", "-"):
            op = self._current().value
            self.pos += 1
            right = self._parse_mul()
            node = BinaryOpNode(op, node, right)
        return node

    def _parse_mul(self) -> ASTNode:
        node = self._parse_power()
        while self._current().type == TokenType.OP and self._current().value in ("*", "/"):
            op = self._current().value
            self.pos += 1
            right = self._parse_power()
            node = BinaryOpNode(op, node, right)
        return node

    def _parse_power(self) -> ASTNode:
        node = self._parse_unary()
        if self._current().type == TokenType.OP and self._current().value == "^":
            self.pos += 1
            right = self._parse_unary()
            node = BinaryOpNode("^", node, right)
        return node

    def _parse_unary(self) -> ASTNode:
        if self._current().type == TokenType.OP and self._current().value == "NEG":
            self.pos += 1
            operand = self._parse_unary()
            return UnaryOpNode("-", operand)
        if self._current().type == TokenType.OP and self._current().value == "!":
            self.pos += 1
            operand = self._parse_unary()
            return UnaryOpNode("!", operand)
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        tok = self._current()

        if tok.type == TokenType.NUMBER:
            self.pos += 1
            return NumberNode(tok.value)

        if tok.type == TokenType.IDENT:
            name = tok.value
            self.pos += 1
            # Check if it's a function call
            if self._current().type == TokenType.LPAREN:
                self.pos += 1  # skip (
                args: list[ASTNode] = []
                if self._current().type != TokenType.RPAREN:
                    args.append(self._parse_ternary())
                    while self._current().type == TokenType.COMMA:
                        self.pos += 1
                        args.append(self._parse_ternary())
                self._eat(TokenType.RPAREN)
                return FunctionCallNode(name, args)
            return IdentNode(name)

        if tok.type == TokenType.LPAREN:
            self.pos += 1
            node = self._parse_ternary()
            self._eat(TokenType.RPAREN)
            return node

        raise SyntaxError(f"Unexpected token: {tok}")


# ---------------------------------------------------------------------------
# Compiler — AST → callable function
# ---------------------------------------------------------------------------

# Data fields that resolve to price matrices
DATA_FIELDS = {
    "open", "close", "high", "low", "volume", "vwap", "returns", "adv20", "sharesout",
}

# Cross-sectional operator names
CS_OPERATORS = {
    "rank": cs.rank,
    "scale": cs.scale,
    "zscore": cs.zscore,
    "winsorize": cs.winsorize,
    "truncate": cs.truncate,
}

# Time-series operator names (need matrix input)
TS_OPERATORS = {
    "delay": ts.delay,
    "delta": ts.delta,
    "ts_sum": ts.ts_sum,
    "ts_mean": ts.ts_mean,
    "ts_std": ts.ts_std,
    "ts_min": ts.ts_min,
    "ts_max": ts.ts_max,
    "ts_rank": ts.ts_rank,
    "ts_skewness": ts.ts_skewness,
    "ts_kurtosis": ts.ts_kurtosis,
    "decay_linear": ts.decay_linear,
    "decay_exp": ts.decay_exp,
    "product": ts.product,
    "count_nans": ts.count_nans,
    "stddev": ts.ts_std,
    "sum": ts.ts_sum,
}

# Two-input time-series operators
TS_TWO_INPUT = {
    "correlation": ts.correlation,
    "covariance": ts.covariance,
    "ts_moment": ts.ts_moment,
}

# Element-wise operators
EW_OPERATORS = {
    "abs": ew.op_abs,
    "sign": ew.sign,
    "log": ew.log,
    "signed_power": ew.signed_power,
    "pasteurize": ew.pasteurize,
    "tail": ew.tail,
}

# Neutralization
NEUTRALIZE_NAMES = {"ind_neutralize", "IndNeutralize"}


class AlphaExpressionCompiler:
    """Compile an AST node into a callable evaluation function."""

    def __init__(self, universe: str = "TOP3000", lookback: int = 256):
        self.universe = universe
        self.lookback = lookback

    def compile(self, node: ASTNode) -> Callable:
        """
        Compile AST → callable function.

        Returns: (date, ctx) -> pd.Series  (ticker -> value)
        """
        def executor(date: dt.date, ctx: Any) -> dict[str, float]:
            result = self._evaluate(node, date, ctx)
            if isinstance(result, pd.Series):
                return result.to_dict()
            elif isinstance(result, (int, float)):
                # Broadcast scalar to universe
                universe = ctx.get_universe(date, self.universe)
                return {t: float(result) for t in universe}
            return result

        return executor

    def _evaluate(self, node: ASTNode, date: dt.date, ctx: Any) -> Any:
        if isinstance(node, NumberNode):
            return node.value

        if isinstance(node, IdentNode):
            # Data field reference — return the current cross-section
            if node.name in DATA_FIELDS:
                mat = ctx.get_matrix(node.name, date, self.lookback, self.universe)
                if mat.empty:
                    return pd.Series(dtype=float)
                return mat.iloc[-1]  # latest cross-section
            # String literal (e.g., industry name for ind_neutralize)
            return node.name

        if isinstance(node, UnaryOpNode):
            operand = self._evaluate(node.operand, date, ctx)
            if node.op == "-":
                if isinstance(operand, pd.Series):
                    return -operand
                return -operand
            if node.op == "!":
                if isinstance(operand, pd.Series):
                    return (~operand.astype(bool)).astype(float)
                return float(not operand)

        if isinstance(node, BinaryOpNode):
            left = self._evaluate(node.left, date, ctx)
            right = self._evaluate(node.right, date, ctx)
            return self._binary_op(node.op, left, right)

        if isinstance(node, TernaryNode):
            cond = self._evaluate(node.condition, date, ctx)
            true_val = self._evaluate(node.true_expr, date, ctx)
            false_val = self._evaluate(node.false_expr, date, ctx)
            if isinstance(cond, pd.Series):
                return true_val.where(cond.astype(bool), false_val)
            return true_val if cond else false_val

        if isinstance(node, FunctionCallNode):
            return self._call_function(node, date, ctx)

        raise ValueError(f"Unknown node type: {type(node)}")

    def _binary_op(self, op: str, left: Any, right: Any) -> Any:
        """Evaluate a binary operation."""
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if not isinstance(b, (int, float)) or b != 0 else np.nan,
            "^": lambda a, b: a ** b,
            "<": lambda a, b: (a < b).astype(float) if isinstance(a, pd.Series) else float(a < b),
            "<=": lambda a, b: (a <= b).astype(float) if isinstance(a, pd.Series) else float(a <= b),
            ">": lambda a, b: (a > b).astype(float) if isinstance(a, pd.Series) else float(a > b),
            ">=": lambda a, b: (a >= b).astype(float) if isinstance(a, pd.Series) else float(a >= b),
            "==": lambda a, b: (a == b).astype(float) if isinstance(a, pd.Series) else float(a == b),
            "!=": lambda a, b: (a != b).astype(float) if isinstance(a, pd.Series) else float(a != b),
            "||": lambda a, b: ((a.astype(bool) | b.astype(bool)).astype(float) if isinstance(a, pd.Series) else float(a or b)),
            "&&": lambda a, b: ((a.astype(bool) & b.astype(bool)).astype(float) if isinstance(a, pd.Series) else float(a and b)),
        }
        fn = ops.get(op)
        if fn is None:
            raise ValueError(f"Unknown operator: {op}")

        # Handle division by zero for Series
        if op == "/":
            if isinstance(right, pd.Series):
                return left / right.replace(0, np.nan)
            elif isinstance(right, (int, float)) and right == 0:
                if isinstance(left, pd.Series):
                    return pd.Series(np.nan, index=left.index)
                return np.nan

        return fn(left, right)

    def _call_function(self, node: FunctionCallNode, date: dt.date, ctx: Any) -> Any:
        """Evaluate a function call."""
        name = node.name

        # --- Cross-sectional operators ---
        if name in CS_OPERATORS:
            arg = self._evaluate(node.args[0], date, ctx)
            if not isinstance(arg, pd.Series):
                return arg
            if name == "winsorize" and len(node.args) >= 2:
                limits_val = self._evaluate(node.args[1], date, ctx)
                return CS_OPERATORS[name](arg, limits_val)
            return CS_OPERATORS[name](arg)

        # --- Neutralization ---
        if name in NEUTRALIZE_NAMES:
            alpha = self._evaluate(node.args[0], date, ctx)
            if not isinstance(alpha, pd.Series):
                return alpha
            if len(node.args) < 2:
                return alpha
            group_name = node.args[1]
            if isinstance(group_name, IdentNode):
                level = group_name.name  # "industry", "subindustry", "sector"
            else:
                level = str(self._evaluate(group_name, date, ctx))
            # Build groups series
            groups = pd.Series(
                {t: ctx.get_industry(t, date, level=level) for t in alpha.index}
            )
            return cs.ind_neutralize(alpha, groups)

        # --- Time-series operators (single input) ---
        if name in TS_OPERATORS:
            # First arg is a data field or expression, second is window
            first_arg = node.args[0]
            # Get the matrix for the first arg
            mat = self._get_matrix_for_node(first_arg, date, ctx)
            n = int(self._evaluate(node.args[1], date, ctx)) if len(node.args) > 1 else 10
            return TS_OPERATORS[name](mat, n)

        # --- Two-input time-series operators ---
        if name in TS_TWO_INPUT:
            mat_x = self._get_matrix_for_node(node.args[0], date, ctx)
            mat_y = self._get_matrix_for_node(node.args[1], date, ctx)
            n = int(self._evaluate(node.args[2], date, ctx)) if len(node.args) > 2 else 10
            if name == "ts_moment":
                # ts_moment(x, k, n) — k is the moment order
                k = int(self._evaluate(node.args[1], date, ctx))
                mat = self._get_matrix_for_node(node.args[0], date, ctx)
                n = int(self._evaluate(node.args[2], date, ctx))
                return TS_TWO_INPUT[name](mat, k, n)
            return TS_TWO_INPUT[name](mat_x, mat_y, n)

        # --- Element-wise operators ---
        if name in EW_OPERATORS:
            arg = self._evaluate(node.args[0], date, ctx)
            if name == "signed_power":
                e = self._evaluate(node.args[1], date, ctx)
                return EW_OPERATORS[name](arg, e)
            if name == "tail":
                lower = self._evaluate(node.args[1], date, ctx)
                upper = self._evaluate(node.args[2], date, ctx)
                newval = self._evaluate(node.args[3], date, ctx) if len(node.args) > 3 else 0.0
                return EW_OPERATORS[name](arg, lower, upper, newval)
            return EW_OPERATORS[name](arg)

        # --- Math operators mapped to element-wise ---
        if name == "max" and len(node.args) == 2:
            a = self._evaluate(node.args[0], date, ctx)
            b = self._evaluate(node.args[1], date, ctx)
            return ew.op_max(a, b)
        if name == "min" and len(node.args) == 2:
            a = self._evaluate(node.args[0], date, ctx)
            b = self._evaluate(node.args[1], date, ctx)
            return ew.op_min(a, b)

        raise ValueError(f"Unknown function: {name}")

    def _get_matrix_for_node(self, node: ASTNode, date: dt.date, ctx: Any) -> pd.DataFrame:
        """Get a matrix (dates × tickers) for a node — either a data field or computed."""
        if isinstance(node, IdentNode) and node.name in DATA_FIELDS:
            return ctx.get_matrix(node.name, date, self.lookback, self.universe)

        # For computed expressions, we'd need to evaluate across all lookback dates
        # For now, evaluate the expression for each date in the lookback window
        # This is slower but supports arbitrary nested expressions
        mat = ctx.get_matrix("close", date, self.lookback, self.universe)  # just for index
        if mat.empty:
            return mat

        results: dict[Any, pd.Series] = {}
        for d in mat.index:
            val = self._evaluate(node, d if isinstance(d, dt.date) else d.date(), ctx)
            if isinstance(val, pd.Series):
                results[d] = val
            elif isinstance(val, (int, float)):
                results[d] = pd.Series(val, index=mat.columns)

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AlphaExpressionParser:
    """
    Parse WebSim-style expression strings and compile them into callable Python functions.

    Usage:
        parser = AlphaExpressionParser()
        compute_fn = parser.parse("-rank(delta(close, 5))")
        result = compute_fn(date, ctx)  # Returns dict[str, float]
    """

    def __init__(self, universe: str = "TOP3000", lookback: int = 256):
        self.universe = universe
        self.lookback = lookback

    def parse(self, expression: str) -> Callable:
        """
        Parse an expression string into a callable function.

        The returned function has signature: (date: dt.date, ctx: DataContext) -> dict[str, float]
        """
        lexer = Lexer(expression)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast_node = parser.parse()
        compiler = AlphaExpressionCompiler(self.universe, self.lookback)
        return compiler.compile(ast_node)

    def parse_to_ast(self, expression: str) -> ASTNode:
        """Parse an expression to its AST representation (for debugging)."""
        lexer = Lexer(expression)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()
