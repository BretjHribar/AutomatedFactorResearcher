"""
FastExpression Parser — compiles WorldQuant BRAIN expression strings into
callable Python functions that operate on DataFrames.

This is the vectorized counterpart to the per-date parser.py. Where parser.py
evaluates expressions one date at a time, FastExpression evaluates the ENTIRE
time series at once using vectorized pandas operations — exactly like the
original GPfunctions.py + DEAP workflow.

Supported syntax (matching BRAIN fastexpression exactly):
    rank(ts_delta(divide(close, volume), 120))
    rank(ts_regression(revenue, assets, 220, lag=110, rettype=2))
    rank(ts_zscore(close / enterprise_value, 120)) * rank(ts_rank(returns, 60))
    group_neutralize(rank(-delta(close, 5)), subindustry)
    -rank(delta(close, 5))
    close > 0 ? rank(close) : -rank(close)

Usage:
    engine = FastExpressionEngine(data_fields={'close': df_close, 'volume': df_volume, ...})
    alpha_df = engine.evaluate("rank(ts_delta(divide(close, volume), 120))")
    # Returns DataFrame (dates × tickers) of alpha values
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.operators import vectorized as ops


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class _TT:
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    OP = "OP"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    QUESTION = "QUESTION"
    COLON = "COLON"
    EQUALS = "EQUALS"
    EOF = "EOF"


class _Token:
    __slots__ = ("type", "value")

    def __init__(self, type_: str, value: Any):
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class FastExpressionLexer:
    """Tokenize a BRAIN fastexpression string."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def tokenize(self) -> list[_Token]:
        tokens: list[_Token] = []
        while self.pos < len(self.text):
            ch = self.text[self.pos]

            if ch.isspace():
                self.pos += 1
                continue

            if ch == '(':
                tokens.append(_Token(_TT.LPAREN, "("))
                self.pos += 1
            elif ch == ')':
                tokens.append(_Token(_TT.RPAREN, ")"))
                self.pos += 1
            elif ch == ',':
                tokens.append(_Token(_TT.COMMA, ","))
                self.pos += 1
            elif ch == '?':
                tokens.append(_Token(_TT.QUESTION, "?"))
                self.pos += 1
            elif ch == ':':
                tokens.append(_Token(_TT.COLON, ":"))
                self.pos += 1
            elif ch == '=' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '=':
                tokens.append(_Token(_TT.OP, "=="))
                self.pos += 2
            elif ch == '=':
                tokens.append(_Token(_TT.EQUALS, "="))
                self.pos += 1
            elif ch in '0123456789' or (ch == '.' and self._peek_digit()):
                tokens.append(self._read_number())
            elif ch == '-' and self._is_unary_minus(tokens):
                if self.pos + 1 < len(self.text) and (self.text[self.pos + 1].isdigit() or self.text[self.pos + 1] == '.'):
                    self.pos += 1
                    num_tok = self._read_number()
                    num_tok.value = -num_tok.value
                    tokens.append(num_tok)
                else:
                    tokens.append(_Token(_TT.OP, "NEG"))
                    self.pos += 1
            elif self._match_two_char_op():
                tokens.append(_Token(_TT.OP, self.text[self.pos:self.pos + 2]))
                self.pos += 2
            elif ch in "+-*/^<>!":
                tokens.append(_Token(_TT.OP, ch))
                self.pos += 1
            elif ch.isalpha() or ch == '_':
                tokens.append(self._read_ident())
            else:
                raise SyntaxError(f"Unexpected character {ch!r} at position {self.pos}")

        tokens.append(_Token(_TT.EOF, None))
        return tokens

    def _peek_digit(self) -> bool:
        return self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()

    def _read_number(self) -> _Token:
        start = self.pos
        has_dot = False
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            elif c.isdigit():
                self.pos += 1
            else:
                break
        # Handle scientific notation: e.g. 1e-10, 1e10, 1E+5
        if self.pos < len(self.text) and self.text[self.pos] in ('e', 'E'):
            self.pos += 1  # consume 'e'/'E'
            if self.pos < len(self.text) and self.text[self.pos] in ('+', '-'):
                self.pos += 1  # consume sign
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        return _Token(_TT.NUMBER, float(self.text[start:self.pos]))

    def _read_ident(self) -> _Token:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        return _Token(_TT.IDENT, self.text[start:self.pos])

    def _match_two_char_op(self) -> bool:
        if self.pos + 1 >= len(self.text):
            return False
        two = self.text[self.pos:self.pos + 2]
        return two in ("||", "&&", "!=", "<=", ">=")

    def _is_unary_minus(self, tokens: list[_Token]) -> bool:
        if not tokens:
            return True
        last = tokens[-1]
        return last.type in (_TT.OP, _TT.LPAREN, _TT.COMMA, _TT.QUESTION, _TT.COLON, _TT.EQUALS)


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

class _ASTNode:
    pass

class _NumberNode(_ASTNode):
    __slots__ = ("value",)
    def __init__(self, value: float):
        self.value = value

class _IdentNode(_ASTNode):
    __slots__ = ("name",)
    def __init__(self, name: str):
        self.name = name

class _UnaryOpNode(_ASTNode):
    __slots__ = ("op", "operand")
    def __init__(self, op: str, operand: _ASTNode):
        self.op = op
        self.operand = operand

class _BinaryOpNode(_ASTNode):
    __slots__ = ("op", "left", "right")
    def __init__(self, op: str, left: _ASTNode, right: _ASTNode):
        self.op = op
        self.left = left
        self.right = right

class _FunctionCallNode(_ASTNode):
    __slots__ = ("name", "args", "kwargs")
    def __init__(self, name: str, args: list[_ASTNode], kwargs: dict[str, _ASTNode]):
        self.name = name
        self.args = args
        self.kwargs = kwargs

class _TernaryNode(_ASTNode):
    __slots__ = ("condition", "true_expr", "false_expr")
    def __init__(self, condition: _ASTNode, true_expr: _ASTNode, false_expr: _ASTNode):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# ---------------------------------------------------------------------------

class FastExpressionParser:
    """Parse fastexpression tokens into an AST."""

    def __init__(self, tokens: list[_Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> _Token:
        return self.tokens[self.pos]

    def _eat(self, type_: str) -> _Token:
        tok = self._current()
        if tok.type != type_:
            raise SyntaxError(f"Expected {type_}, got {tok}")
        self.pos += 1
        return tok

    def parse(self) -> _ASTNode:
        node = self._parse_ternary()
        if self._current().type != _TT.EOF:
            raise SyntaxError(f"Unexpected trailing token: {self._current()}")
        return node

    def _parse_ternary(self) -> _ASTNode:
        node = self._parse_or()
        if self._current().type == _TT.QUESTION:
            self.pos += 1
            true_expr = self._parse_ternary()
            self._eat(_TT.COLON)
            false_expr = self._parse_ternary()
            return _TernaryNode(node, true_expr, false_expr)
        return node

    def _parse_or(self) -> _ASTNode:
        node = self._parse_and()
        while self._current().type == _TT.OP and self._current().value == "||":
            self.pos += 1
            node = _BinaryOpNode("||", node, self._parse_and())
        return node

    def _parse_and(self) -> _ASTNode:
        node = self._parse_comparison()
        while self._current().type == _TT.OP and self._current().value == "&&":
            self.pos += 1
            node = _BinaryOpNode("&&", node, self._parse_comparison())
        return node

    def _parse_comparison(self) -> _ASTNode:
        node = self._parse_add()
        while self._current().type == _TT.OP and self._current().value in ("<", "<=", ">", ">=", "==", "!="):
            op = self._current().value
            self.pos += 1
            node = _BinaryOpNode(op, node, self._parse_add())
        return node

    def _parse_add(self) -> _ASTNode:
        node = self._parse_mul()
        while self._current().type == _TT.OP and self._current().value in ("+", "-"):
            op = self._current().value
            self.pos += 1
            node = _BinaryOpNode(op, node, self._parse_mul())
        return node

    def _parse_mul(self) -> _ASTNode:
        node = self._parse_power()
        while self._current().type == _TT.OP and self._current().value in ("*", "/"):
            op = self._current().value
            self.pos += 1
            node = _BinaryOpNode(op, node, self._parse_power())
        return node

    def _parse_power(self) -> _ASTNode:
        node = self._parse_unary()
        if self._current().type == _TT.OP and self._current().value == "^":
            self.pos += 1
            node = _BinaryOpNode("^", node, self._parse_unary())
        return node

    def _parse_unary(self) -> _ASTNode:
        if self._current().type == _TT.OP and self._current().value == "NEG":
            self.pos += 1
            return _UnaryOpNode("-", self._parse_unary())
        if self._current().type == _TT.OP and self._current().value == "!":
            self.pos += 1
            return _UnaryOpNode("!", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> _ASTNode:
        tok = self._current()

        if tok.type == _TT.NUMBER:
            self.pos += 1
            return _NumberNode(tok.value)

        if tok.type == _TT.IDENT:
            name = tok.value
            self.pos += 1
            if self._current().type == _TT.LPAREN:
                return self._parse_function_call(name)
            return _IdentNode(name)

        if tok.type == _TT.LPAREN:
            self.pos += 1
            node = self._parse_ternary()
            self._eat(_TT.RPAREN)
            return node

        raise SyntaxError(f"Unexpected token: {tok}")

    def _parse_function_call(self, name: str) -> _FunctionCallNode:
        """Parse function call with positional and keyword arguments."""
        self._eat(_TT.LPAREN)
        args: list[_ASTNode] = []
        kwargs: dict[str, _ASTNode] = {}

        if self._current().type != _TT.RPAREN:
            self._parse_argument(args, kwargs)
            while self._current().type == _TT.COMMA:
                self.pos += 1
                self._parse_argument(args, kwargs)

        self._eat(_TT.RPAREN)
        return _FunctionCallNode(name, args, kwargs)

    def _parse_argument(self, args: list[_ASTNode], kwargs: dict[str, _ASTNode]) -> None:
        """Parse a single argument — could be positional or keyword (name=value)."""
        # Check for keyword arg:  ident = expr
        if (self._current().type == _TT.IDENT
                and self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1].type == _TT.EQUALS):
            key = self._current().value
            self.pos += 1  # skip ident
            self.pos += 1  # skip =
            kwargs[key] = self._parse_ternary()
        else:
            args.append(self._parse_ternary())


# ---------------------------------------------------------------------------
# Evaluator — compiles AST and evaluates against data
# ---------------------------------------------------------------------------

# Map BRAIN operator names → vectorized function + metadata
# (func, n_positional_args, accepts_kwargs)
_OPERATOR_REGISTRY: dict[str, tuple[Callable, list[str]]] = {}


def _register(name: str, func: Callable, params: list[str] | None = None) -> None:
    """Register an operator name to its implementation."""
    _OPERATOR_REGISTRY[name] = (func, params or [])


# --- Time-series operators ---
_register("ts_sum",       ops.ts_sum,       ["df", "window"])
_register("sma",          ops.sma,          ["df", "window"])
_register("ts_mean",      ops.ts_mean,      ["df", "window"])
_register("ts_rank",      ops.ts_rank,      ["df", "window"])
_register("ts_min",       ops.ts_min,       ["df", "window"])
_register("ts_max",       ops.ts_max,       ["df", "window"])
_register("delta",        ops.delta,        ["df", "period"])
_register("ts_delta",     ops.ts_delta,     ["df", "period"])
_register("stddev",       ops.stddev,       ["df", "window"])
_register("ts_std_dev",   ops.ts_std_dev,   ["df", "window"])
_register("correlation",  ops.correlation,  ["x", "y", "window"])
_register("ts_corr",      ops.ts_corr,      ["x", "y", "window"])
_register("covariance",   ops.covariance,   ["x", "y", "window"])
_register("ts_cov",       ops.ts_cov,       ["x", "y", "window"])
_register("Product",      ops.Product,      ["df", "window"])
_register("delay",        ops.delay,        ["df", "period"])
_register("ts_delay",     ops.ts_delay,     ["df", "period"])
_register("ArgMax",       ops.ArgMax,       ["df", "window"])
_register("ts_argmax",    ops.ts_argmax,    ["df", "window"])
_register("ArgMin",       ops.ArgMin,       ["df", "window"])
_register("ts_argmin",    ops.ts_argmin,    ["df", "window"])
_register("ts_skewness",  ops.ts_skewness,  ["df", "window"])
_register("ts_kurtosis",  ops.ts_kurtosis,  ["df", "window"])
_register("ts_zscore",    ops.ts_zscore,    ["df", "window"])
_register("ts_av_diff",   ops.ts_av_diff,   ["df", "window"])
_register("ts_regression", ops.ts_regression, ["y", "x", "window", "lag", "rettype"])
_register("hump",         ops.hump,         ["df", "hump_val"])
_register("ts_entropy",   ops.ts_entropy,   ["df", "window"])
_register("ts_count_nans", ops.ts_count_nans, ["df", "window"])

# --- Decay ---
_register("Decay_lin",    ops.Decay_lin,    ["df", "window"])
_register("decay_linear", ops.decay_linear, ["df", "window"])
_register("Decay_exp",    ops.Decay_exp,    ["df", "alpha_exp"])
_register("decay_exp",    ops.decay_exp,    ["df", "alpha_exp"])

# --- Cross-sectional operators ---
_register("rank",           ops.rank,           ["df"])
_register("scale",          ops.scale,          ["df", "k"])
_register("zscore",         ops.zscore,         ["df"])
_register("zscore_cs",      ops.zscore_cs,      ["df"])
_register("winsorize",      ops.winsorize,      ["df"])
_register("truncate",       ops.truncate,       ["df", "max_weight"])

# --- Element-wise operators ---
_register("add",            ops.add,            ["left", "right"])
_register("subtract",       ops.subtract,       ["left", "right"])
_register("multiply",       ops.multiply,       ["left", "right"])
_register("mul",            ops.multiply,       ["left", "right"])
_register("divide",         ops.divide,         ["left", "right"])
_register("div",            ops.divide,         ["left", "right"])
_register("true_divide",    ops.true_divide,    ["left", "right"])
_register("protectedDiv",   ops.protectedDiv,   ["left", "right"])
_register("negative",       ops.negative,       ["df"])
_register("Abs",            ops.Abs,            ["df"])
_register("abs",            ops.abs_op,         ["df"])
_register("Sign",           ops.Sign,           ["df"])
_register("sign",           ops.sign,           ["df"])
_register("SignedPower",    ops.SignedPower,     ["df", "y"])
_register("signed_power",   ops.signed_power,   ["df", "y"])
_register("Inverse",        ops.Inverse,        ["df"])
_register("inverse",        ops.inverse,        ["df"])
_register("log",            ops.log,            ["df"])
_register("log10",          ops.log10,          ["df"])
_register("sqrt",           ops.sqrt,           ["df"])
_register("square",         ops.square,         ["df"])
_register("log_diff",       ops.log_diff,       ["df"])
_register("s_log_1p",       ops.s_log_1p,       ["df"])
_register("Tail",           ops.Tail,           ["df", "cutoff"])
_register("tail",           ops.tail,           ["df", "cutoff"])
_register("df_max",         ops.df_max,         ["left", "right"])
_register("df_min",         ops.df_min,         ["left", "right"])
_register("max",            ops.df_max,         ["left", "right"])
_register("min",            ops.df_min,         ["left", "right"])
_register("if_else",        ops.if_else,        ["cond", "true_val", "false_val"])
_register("power",          ops.power,          ["df", "exp"])
_register("pasteurize",     ops.pasteurize,     ["df"])

# Scalar-DataFrame arithmetic
_register("npfadd",         ops.npfadd,         ["df", "f"])
_register("npfsub",         ops.npfsub,         ["df", "f"])
_register("npfmul",         ops.npfmul,         ["df", "f"])
_register("npfdiv",         ops.npfdiv,         ["df", "f"])

# group operators — handled specially because they need groups data
# registered without group arg; evaluated with special logic
_register("group_neutralize", None, ["df", "group_level"])
_register("group_rank",      None, ["df", "group_level"])
_register("IndNeutralize",   None, ["df", "group_level"])
_register("market_neutralize", ops.market_neutralize, ["df"])
_register("group_mean",       None, ["df", "weight", "group_level"])
_register("group_scale",      None, ["df", "group_level"])
_register("group_zscore",     None, ["df", "group_level"])
_register("group_backfill",   None, ["df", "group_level", "window"])

# --- Additional BRAIN operators ---
_register("normalize",        ops.normalize,        ["df"])
_register("quantile",         ops.quantile,         ["df"])
_register("ts_backfill",      ops.ts_backfill,      ["df", "window", "k"])
_register("ts_quantile",      ops.ts_quantile,      ["df", "window"])
_register("ts_scale",         ops.ts_scale,         ["df", "window", "constant"])
_register("ts_product",       ops.ts_product,       ["df", "window"])
_register("kth_element",      ops.kth_element,      ["df", "window", "k"])
_register("last_diff_value",  ops.last_diff_value,  ["df", "window"])
_register("ts_step",          ops.ts_step,          ["constant"])
_register("days_from_last_change", ops.days_from_last_change, ["df"])
_register("reverse",          ops.reverse,          ["df"])
_register("is_nan",           ops.is_nan,           ["df"])
_register("trade_when",       ops.trade_when,       ["cond", "alpha", "fallback"])
_register("bucket",           ops.bucket,           ["df", "buckets"])
_register("vec_avg",          ops.vec_avg,          ["df"])
_register("vec_sum",          ops.vec_sum,          ["df"])
_register("ts_decay_linear",  ops.ts_decay_linear,  ["df", "window"])
_register("ts_arg_max",       ops.ts_arg_max,       ["df", "window"])
_register("ts_arg_min",       ops.ts_arg_min,       ["df", "window"])
_register("ts_covariance",    ops.ts_covariance,    ["y", "x", "window"])
_register("not",              ops.reverse,          ["df"])  # logical not ≈ negate for numeric
_register("extend",           ops.extend,           ["i"])   # identity / type coercion from GP


# Group level identifiers (not data fields)
_GROUP_LEVELS = {"subindustry", "industry", "sector", "country", "market", "exchange"}


class FastExpressionEngine:
    """
    Compile and evaluate BRAIN fastexpression strings.

    Usage:
        engine = FastExpressionEngine(
            data_fields={'close': df_close, 'volume': df_volume, ...},
            groups={'industry': industry_series, ...}
        )
        result = engine.evaluate("rank(ts_delta(divide(close, volume), 120))")
    """

    def __init__(
        self,
        data_fields: dict[str, pd.DataFrame] | None = None,
        groups: dict[str, pd.Series] | None = None,
    ):
        """
        Args:
            data_fields: Mapping of field name → DataFrame (dates × tickers).
                           e.g., {'close': df_close, 'volume': df_volume, ...}
            groups: Mapping of level → Series (ticker → group_label).
                      e.g., {'industry': pd.Series({'AAPL': 'Tech', ...})}
        """
        self.data_fields: dict[str, pd.DataFrame] = data_fields or {}
        self.groups: dict[str, pd.Series] = groups or {}

    def add_field(self, name: str, df: pd.DataFrame) -> None:
        """Add or update a data field."""
        self.data_fields[name] = df

    def add_group(self, level: str, groups: pd.Series) -> None:
        """Add or update group classifications."""
        self.groups[level] = groups

    def evaluate(self, expression: str) -> pd.DataFrame:
        """
        Parse and evaluate a fastexpression string.

        Returns: DataFrame (dates × tickers) of alpha values.
        """
        lexer = FastExpressionLexer(expression)
        tokens = lexer.tokenize()
        parser = FastExpressionParser(tokens)
        ast = parser.parse()
        return self._eval(ast)

    def compile(self, expression: str) -> Callable[[], pd.DataFrame]:
        """
        Parse an expression and return a callable that evaluates it.

        Useful for repeated evaluation with different data.
        """
        lexer = FastExpressionLexer(expression)
        tokens = lexer.tokenize()
        parser = FastExpressionParser(tokens)
        ast = parser.parse()
        return lambda: self._eval(ast)

    def parse_to_ast(self, expression: str) -> _ASTNode:
        """Parse an expression string to its AST (for debugging)."""
        lexer = FastExpressionLexer(expression)
        tokens = lexer.tokenize()
        parser = FastExpressionParser(tokens)
        return parser.parse()

    # --- Internal evaluation ---

    def _eval(self, node: _ASTNode) -> Any:
        if isinstance(node, _NumberNode):
            return node.value

        if isinstance(node, _IdentNode):
            name = node.name
            # Check if it's a registered data field
            if name in self.data_fields:
                return self.data_fields[name]
            # Check if it's a group level identifier
            if name in _GROUP_LEVELS:
                return name  # Return as string for group operators
            # Could be a constant name
            raise ValueError(f"Unknown identifier: '{name}'. "
                             f"Available fields: {list(self.data_fields.keys())}")

        if isinstance(node, _UnaryOpNode):
            operand = self._eval(node.operand)
            if node.op == "-":
                if isinstance(operand, pd.DataFrame):
                    return -operand
                return -operand
            if node.op == "!":
                if isinstance(operand, pd.DataFrame):
                    return (~operand.astype(bool)).astype(float)
                return float(not operand)
            raise ValueError(f"Unknown unary op: {node.op}")

        if isinstance(node, _BinaryOpNode):
            left = self._eval(node.left)
            right = self._eval(node.right)
            return self._binary_op(node.op, left, right)

        if isinstance(node, _TernaryNode):
            cond = self._eval(node.condition)
            true_val = self._eval(node.true_expr)
            false_val = self._eval(node.false_expr)
            if isinstance(cond, pd.DataFrame):
                return true_val.where(cond.astype(bool), false_val)
            return true_val if cond else false_val

        if isinstance(node, _FunctionCallNode):
            return self._call_function(node)

        raise ValueError(f"Unknown AST node type: {type(node)}")

    def _binary_op(self, op: str, left: Any, right: Any) -> Any:
        """Evaluate binary operations."""
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if isinstance(right, pd.DataFrame):
                return left / right.replace(0, np.nan)
            elif isinstance(right, (int, float)):
                return left / right if right != 0 else (left * np.nan if isinstance(left, pd.DataFrame) else np.nan)
            return left / right
        if op == "^":
            return left ** right
        if op == "<":
            return (left < right).astype(float) if isinstance(left, pd.DataFrame) else float(left < right)
        if op == "<=":
            return (left <= right).astype(float) if isinstance(left, pd.DataFrame) else float(left <= right)
        if op == ">":
            return (left > right).astype(float) if isinstance(left, pd.DataFrame) else float(left > right)
        if op == ">=":
            return (left >= right).astype(float) if isinstance(left, pd.DataFrame) else float(left >= right)
        if op == "==":
            return (left == right).astype(float) if isinstance(left, pd.DataFrame) else float(left == right)
        if op == "!=":
            return (left != right).astype(float) if isinstance(left, pd.DataFrame) else float(left != right)
        if op == "||":
            if isinstance(left, pd.DataFrame):
                return (left.astype(bool) | right.astype(bool)).astype(float)
            return float(left or right)
        if op == "&&":
            if isinstance(left, pd.DataFrame):
                return (left.astype(bool) & right.astype(bool)).astype(float)
            return float(left and right)
        raise ValueError(f"Unknown operator: {op}")

    def _call_function(self, node: _FunctionCallNode) -> Any:
        """Evaluate a function call."""
        name = node.name

        # --- Special handling for group operators ---
        if name in ("group_neutralize", "IndNeutralize"):
            return self._eval_group_neutralize(node)
        if name == "group_rank":
            return self._eval_group_rank(node)
        if name == "group_zscore":
            return self._eval_group_op(node, ops.group_zscore)
        if name == "group_scale":
            return self._eval_group_op(node, ops.group_scale)
        if name == "group_mean":
            return self._eval_group_mean(node)
        if name == "group_backfill":
            return self._eval_group_backfill(node)

        # --- Look up in registry ---
        if name not in _OPERATOR_REGISTRY:
            raise ValueError(f"Unknown function: '{name}'. "
                             f"Available: {sorted(_OPERATOR_REGISTRY.keys())}")

        func, param_names = _OPERATOR_REGISTRY[name]

        # Evaluate positional args
        eval_args = [self._eval(a) for a in node.args]

        # Evaluate keyword args
        eval_kwargs = {k: self._eval(v) for k, v in node.kwargs.items()}

        # Convert numeric args to proper types
        call_args = []
        for i, arg in enumerate(eval_args):
            if i < len(param_names):
                pname = param_names[i]
                # Convert float to int for integer params
                if pname in ("window", "period", "lag", "rettype", "k") and isinstance(arg, float):
                    arg = int(arg)
            call_args.append(arg)

        # Convert kwargs numeric values
        for k, v in eval_kwargs.items():
            if k in ("window", "period", "lag", "rettype") and isinstance(v, float):
                eval_kwargs[k] = int(v)

        return func(*call_args, **eval_kwargs)

    def _eval_group_level_arg(self, node: _ASTNode) -> Any:
        """Resolve a group argument without confusing it for a data field.

        Group matrices are also available as data fields for legacy alphas, so
        `group_rank(x, subindustry)` needs this special path: the second
        argument means the group level name, not the encoded parquet matrix.
        """
        if isinstance(node, _IdentNode) and node.name in _GROUP_LEVELS:
            return node.name
        return self._eval(node)

    def _eval_group_neutralize(self, node: _FunctionCallNode) -> pd.DataFrame:
        """Evaluate group_neutralize / IndNeutralize."""
        if len(node.args) < 2:
            raise ValueError("group_neutralize requires at least 2 arguments: (alpha, group_level)")

        alpha_df = self._eval(node.args[0])
        group_level = self._eval_group_level_arg(node.args[1])

        if isinstance(group_level, str) and group_level in self.groups:
            return ops.group_neutralize(alpha_df, self.groups[group_level])
        else:
            raise ValueError(f"Unknown group level: '{group_level}'. "
                             f"Available: {list(self.groups.keys())}")

    def _eval_group_rank(self, node: _FunctionCallNode) -> pd.DataFrame:
        """Evaluate group_rank."""
        if len(node.args) < 2:
            raise ValueError("group_rank requires at least 2 arguments: (alpha, group_level)")

        alpha_df = self._eval(node.args[0])
        group_level = self._eval_group_level_arg(node.args[1])

        if isinstance(group_level, str) and group_level in self.groups:
            return ops.group_rank(alpha_df, self.groups[group_level])
        else:
            raise ValueError(f"Unknown group level: '{group_level}'. "
                             f"Available: {list(self.groups.keys())}")

    def _eval_group_op(self, node: _FunctionCallNode, func) -> pd.DataFrame:
        """Generic evaluate for group_zscore, group_scale, etc."""
        if len(node.args) < 2:
            raise ValueError(f"{node.name} requires (alpha, group_level)")
        alpha_df = self._eval(node.args[0])
        group_level = self._eval_group_level_arg(node.args[1])
        if isinstance(group_level, str) and group_level in self.groups:
            return func(alpha_df, self.groups[group_level])
        raise ValueError(f"Unknown group level: '{group_level}'")

    def _eval_group_mean(self, node: _FunctionCallNode) -> pd.DataFrame:
        """Evaluate group_mean(x, weight, group)."""
        if len(node.args) < 3:
            raise ValueError("group_mean requires (alpha, weight, group_level)")
        alpha_df = self._eval(node.args[0])
        weight_df = self._eval(node.args[1])
        group_level = self._eval_group_level_arg(node.args[2])
        if isinstance(group_level, str) and group_level in self.groups:
            return ops.group_mean(alpha_df, self.groups[group_level], weight_df)
        raise ValueError(f"Unknown group level: '{group_level}'")

    def _eval_group_backfill(self, node: _FunctionCallNode) -> pd.DataFrame:
        """Evaluate group_backfill(x, group, d)."""
        if len(node.args) < 3:
            raise ValueError("group_backfill requires (alpha, group_level, window)")
        alpha_df = self._eval(node.args[0])
        group_level = self._eval_group_level_arg(node.args[1])
        window = int(self._eval(node.args[2]))
        if isinstance(group_level, str) and group_level in self.groups:
            return ops.group_backfill(alpha_df, self.groups[group_level], window)
        raise ValueError(f"Unknown group level: '{group_level}'")


# ---------------------------------------------------------------------------
# Convenience: load engine from InMemoryDataContext
# ---------------------------------------------------------------------------

def create_engine_from_context(ctx: Any) -> FastExpressionEngine:
    """
    Create a FastExpressionEngine by extracting all DataFrames from an
    InMemoryDataContext.

    This bridges the DataContext abstraction with the vectorized engine.
    """
    engine = FastExpressionEngine()

    # Extract price matrices
    for field_name, matrix in ctx._price_matrices.items():
        engine.add_field(field_name, matrix)

    # Build commonly derived fields
    if "close" in ctx._price_matrices:
        close = ctx._price_matrices["close"]

        # Dollar-weighted fields
        if "volume" in ctx._price_matrices:
            vol = ctx._price_matrices["volume"]
            engine.add_field("dollars_traded", close * vol)
            engine.add_field("dollar_volume", close * vol)

        # Historical volatilities
        if "returns" in ctx._price_matrices:
            ret = ctx._price_matrices["returns"]
            for w in [10, 20, 30, 90]:
                engine.add_field(f"hist_vol_{w}",
                                 ret.rolling(w, min_periods=2).std() * (252 ** 0.5))

    # Build group classifications
    for level in ["sector", "industry", "subindustry"]:
        groups = {}
        for ticker, info in ctx._classifications.items():
            if level in info:
                groups[ticker] = info[level]
        if groups:
            engine.add_group(level, pd.Series(groups))

    return engine
