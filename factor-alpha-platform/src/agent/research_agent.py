"""
LLM Alpha Research Agent — generates, evaluates, and iterates on alpha factors.

This is the autonomous research loop that:
1. Prompts an LLM (Gemini, OpenAI, etc.) to generate fastexpression alpha strings
2. Evaluates them locally using our FastExpression engine + simulation pipeline
3. Records results to the alpha database
4. Feeds history back to the LLM for in-context learning
5. Iterates until target metrics are reached

This replaces WorldQuant's platform — we are the evaluation engine.

Usage:
    agent = AlphaResearchAgent.from_synthetic(n_stocks=200, n_days=500)
    await agent.run(n_trials=60, strategy="Find momentum + value composite alphas")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.data.field_catalog import (
    ALL_GROUPS, format_fields_for_prompt, get_field_names_for_groups,
)
from src.evaluation.pipeline import AlphaEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Operator catalog for prompts
# ---------------------------------------------------------------------------

OPERATOR_CATALOG = """
<Arithmetic>
abs(x) – absolute value
add(x, y) – addition
divide(x, y) – x / y
inverse(x) – 1 / x
log(x) – natural log
max(x, y) – maximum
min(x, y) – minimum
multiply(x, y) – multiplication
power(x, y) – x ^ y
reverse(x) – negate (-x)
sign(x) – sign function
signed_power(x, y) – sign-preserving power
sqrt(x) – square root
subtract(x, y) – subtraction

<Logical>
if_else(cond, a, b) – ternary conditional
is_nan(x) – 1 if NaN else 0
Comparisons: <, <=, ==, !=, >=, >

<Time-Series>
ts_arg_max(x, d) / ts_arg_min(x, d)
ts_av_diff(x, d) – deviation from moving average
ts_backfill(x, d, k=1) – forward-fill NaNs
ts_corr(x, y, d) – rolling correlation
ts_count_nans(x, d) – count NaNs in window
ts_covariance(y, x, d) – rolling covariance
ts_decay_linear(x, d) – linearly-weighted moving average
ts_delay(x, d) – lag value by d days
ts_delta(x, d) – change over d days
ts_mean(x, d) – rolling mean
ts_product(x, d) – rolling product
ts_quantile(x, d) – rolling quantile
ts_rank(x, d) – rolling rank as percentile
ts_scale(x, d) – scale to [0,1] over window
ts_std_dev(x, d) – rolling standard deviation
ts_sum(x, d) – rolling sum
ts_zscore(x, d) – rolling z-score
ts_regression(y, x, d, lag=0, rettype=0) – OLS regression
  rettype: 0=residual, 1=intercept, 2=slope, 3=fitted, 4=SSE, 5=SST, 6=R², 7=MSE, 8=SEβ, 9=SEα
hump(x, hump=0.01) – smooth extremes
kth_element(x, d, k) – k-th lookback element
days_from_last_change(x) – days since last value change

<Cross-Sectional>
normalize(x) – normalize to [-1,1]
quantile(x) – quantile transform
rank(x) – cross-sectional percentile rank
scale(x, scale=1) – scale to target sum
winsorize(x) – clip outliers
zscore(x) – cross-sectional z-score
bucket(x, n) – equal-frequency bucketing

<Group>
group_backfill(x, group, d) – fill NaNs within group
group_mean(x, weight, group) – weighted mean within group
group_neutralize(x, group) – demean within group
group_rank(x, group) – rank within group
group_scale(x, group) – scale within group
group_zscore(x, group) – z-score within group

<Vector>
vec_avg(x) – cross-sectional average (broadcast)
vec_sum(x) – cross-sectional sum (broadcast)
""".strip()


# ---------------------------------------------------------------------------
# LLM Provider abstraction
# ---------------------------------------------------------------------------

class LLMProvider:
    """Abstract interface for LLM calls."""

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "unknown"


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash"):
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise ImportError("google-generativeai package required. pip install google-generativeai")

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY required")

        self._genai.configure(api_key=api_key)
        self._model = self._genai.GenerativeModel(model)
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(self, prompt: str) -> str:
        for attempt in range(5):
            try:
                response = await self._model.generate_content_async(prompt)
                if hasattr(response, 'text') and response.text:
                    return response.text
                # Try candidates fallback
                if hasattr(response, 'candidates') and response.candidates:
                    for cand in response.candidates:
                        if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                            texts = [p.text for p in cand.content.parts if hasattr(p, 'text')]
                            if texts:
                                return "\n".join(texts)
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(min(2 ** attempt, 8))
        raise RuntimeError("LLM generation failed after retries")


class StubLLMProvider(LLMProvider):
    """Stub provider for testing without API keys."""

    def __init__(self, expressions: List[str] | None = None):
        self._expressions = expressions or [
            "rank(ts_delta(divide(close, volume), 60))",
            "rank(ts_rank(returns, 30)) * rank(delta(close, 5))",
            "rank(ts_zscore(close / enterprise_value, 120))",
            "rank(ts_regression(revenue, assets, 120, lag=60, rettype=2))",
            "rank(group_rank(inverse(ts_std_dev(divide(rd_expense, sales), 60)), subindustry))",
        ]
        self._idx = 0

    @property
    def model_name(self) -> str:
        return "stub"

    async def generate(self, prompt: str) -> str:
        expr = self._expressions[self._idx % len(self._expressions)]
        self._idx += 1
        return f"ALPHA: {expr}\nREASONING: Stub expression for testing\nPARAMETERS_JSON: {{}}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_alpha_code(text: str) -> str:
    """Extract expression from LLM output. Handles ALPHA: marker and raw expressions."""
    for line in text.split('\n'):
        if line.strip().startswith('ALPHA:'):
            code = line.split(':', 1)[1].strip()
            return _clean_expression(code)

    # Try to find rank(...) patterns
    ranks = re.findall(r'rank\([^\n]+?\)', text)
    if ranks:
        return _clean_expression(' * '.join(ranks[:3]))

    # Fallback: first line with operators
    for line in text.split('\n'):
        line = line.strip()
        if any(op in line for op in ['rank(', 'ts_', 'group_', 'divide(', 'multiply(']):
            # Truncate leading narrative
            for marker in ['rank(', 'ts_', 'group_', 'divide(', 'multiply(']:
                idx = line.find(marker)
                if idx > 0:
                    line = line[idx:]
                    break
            return _clean_expression(line)

    return ""


def _clean_expression(code: str) -> str:
    """Clean raw expression text into valid fastexpression."""
    code = re.sub(r'[*`]+', '', code)  # remove markdown
    code = re.sub(r'^\s*[-*]\s*', '', code)  # remove bullets
    code = re.sub(r'\n+', ' ', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.strip()
    code = re.sub(r'[.,;]+$', '', code)

    # Fix missing operators between consecutive function calls
    code = re.sub(r'\)\s+rank\(', ') * rank(', code)
    code = re.sub(r'\)\s+ts_', ') * ts_', code)
    code = re.sub(r'\)\s+group_', ') * group_', code)

    # Fix operator name variants
    corrections = {
        r'\bdecay_linear\b': 'ts_decay_linear',
        r'\bdelay\b(?!\()': 'ts_delay',
        r'\bmean\b(?!\()': 'ts_mean',
        r'\bsum\b(?!\()': 'ts_sum',
        r'\bcorr\b': 'ts_corr',
        r'\bstd_dev\b': 'ts_std_dev',
        r'\bstd\b': 'ts_std_dev',
    }
    for pattern, replacement in corrections.items():
        code = re.sub(pattern, replacement, code)

    return code


def _extract_reasoning(text: str) -> str:
    """Extract REASONING section from response."""
    for line in text.split('\n'):
        if line.strip().startswith('REASONING:'):
            return line.split(':', 1)[1].strip()
    return ""


def _extract_parameters(text: str) -> Dict[str, Any]:
    """Extract PARAMETERS_JSON from response."""
    defaults = {
        'decay': 0,
        'delay': 1,
        'neutralization': 'subindustry',
        'truncation': 0.1,
    }
    try:
        if 'PARAMETERS_JSON:' in text:
            after = text.split('PARAMETERS_JSON:', 1)[1].strip()
            # Find balanced JSON
            if '{' in after:
                depth = 0
                chars = []
                for ch in after:
                    chars.append(ch)
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            break
                candidate = ''.join(chars).strip()
                params = json.loads(candidate)
                defaults.update({k: v for k, v in params.items() if k in defaults})
    except Exception:
        pass
    return defaults


def _validate_syntax(code: str) -> bool:
    """Basic syntax validation for fastexpression."""
    if not code or len(code.strip()) < 5:
        return False
    if code.count('(') != code.count(')'):
        return False
    if not re.search(r'\w+\(', code):
        return False
    if re.search(r'\(\s*\)', code):
        return False
    if re.search(r',\s*\)', code):
        return False
    if re.search(r',\s*,', code):
        return False
    return True


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------

class AlphaResearchAgent:
    """
    LLM-driven alpha research agent.

    Generates fastexpression alphas, evaluates them locally, and iterates
    using in-context learning from prior results.
    """

    def __init__(
        self,
        evaluator: AlphaEvaluator,
        llm: LLMProvider | None = None,
        allowed_data_groups: List[str] | None = None,
        data_fields_pct: float = 40.0,
        operators_pct: float = 60.0,
        history_lookback: int = 30,
    ):
        self.evaluator = evaluator
        self.llm = llm or StubLLMProvider()
        self.history_lookback = history_lookback
        self.data_fields_pct = max(10, min(100, data_fields_pct))
        self.operators_pct = max(10, min(100, operators_pct))

        # Configure allowed data groups
        if allowed_data_groups:
            self.allowed_data_groups = {g.upper().strip() for g in allowed_data_groups}
        else:
            self.allowed_data_groups = {
                "MARKET DATA",
                "FINANCIAL STATEMENT DATA",
                "ADDITIONAL FINANCIAL DATA",
                "GROUPING FIELDS",
            }

        # In-run thought history for in-context learning
        self.thought_history: List[Dict[str, Any]] = []

        # Cache scoped operator/field text
        self._scoped_text: str | None = None

    @classmethod
    def from_synthetic(
        cls,
        n_stocks: int = 200,
        n_days: int = 500,
        seed: int = 42,
        llm: LLMProvider | None = None,
        **kwargs,
    ) -> "AlphaResearchAgent":
        """Create agent backed by synthetic data for development/testing."""
        evaluator = AlphaEvaluator.from_synthetic(
            n_stocks=n_stocks, n_days=n_days, seed=seed
        )
        return cls(evaluator=evaluator, llm=llm, **kwargs)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _get_scoped_text(self) -> str:
        """Get scoped operator + field text for prompt (cached per run)."""
        if self._scoped_text is not None:
            return self._scoped_text

        # Operators: sample a subset
        op_lines = [l.strip() for l in OPERATOR_CATALOG.split('\n')
                     if l.strip() and not l.strip().startswith('<')]
        n_ops = max(5, int(len(op_lines) * self.operators_pct / 100))
        sampled_ops = random.sample(op_lines, min(n_ops, len(op_lines)))

        # Data fields: from catalog
        fields_text = format_fields_for_prompt(
            groups=list(self.allowed_data_groups),
            max_per_group=max(5, int(50 * self.data_fields_pct / 100)),
        )

        self._scoped_text = (
            f"OPERATORS ({len(sampled_ops)} sampled):\n"
            + "\n".join(sampled_ops)
            + f"\n\nDATA FIELDS:\n{fields_text}"
        )
        return self._scoped_text

    def _build_thought_block(self) -> str:
        """Format in-run thought history for the prompt."""
        if not self.thought_history:
            return "None yet"
        lines = []
        for t in self.thought_history:
            idx = t.get('trial', '?')
            code = t.get('expression', '')[:80]
            outcome = t.get('outcome', {})
            if outcome.get('success'):
                s = outcome.get('sharpe', 0)
                f = outcome.get('fitness', 0)
                status = f"✓ S={s:.2f} F={f:.2f}"
            else:
                err = (outcome.get('error') or '')[:40]
                status = f"✗ {err}"
            lines.append(f"T{idx}: {status} | {code}")
        return "\n".join(lines)

    def _build_prompt(self, strategy: str, history: List[Dict]) -> str:
        """Build the full prompt for the LLM."""
        # Format history
        hist_lines = []
        for h in history[-self.history_lookback:]:
            hist_lines.append(
                f"  F={h.get('fitness', 0):.2f} S={h.get('sharpe', 0):.2f} | {h['code']}"
            )
        hist_block = "\n".join(hist_lines) if hist_lines else "None yet"

        thought_block = self._build_thought_block()
        scoped_text = self._get_scoped_text()

        prompt = f"""You are an expert quantitative researcher creating equity factor alphas.

TASK: Generate a NOVEL alpha factor expression. {strategy}

TARGET METRICS:
- Sharpe ratio ≥ 1.25 (risk-adjusted returns)
- Fitness ≥ 1.0 (Fitness = Sharpe * sqrt(|returns| / max(turnover, 0.125)))
- Low turnover preferred — use longer lookback periods (60-180 days)
- Positive PnL across time

STRATEGY NOTES:
- Sacrifice Sharpe to increase Fitness by reducing turnover
- Use fundamental data that changes infrequently for lower turnover
- Combine 2-3 orthogonal rank() components multiplied together
- Think about what economic signal each component captures

RECENT RESULTS (avoid duplicates):
{hist_block}

RUN HISTORY ({len(self.thought_history)} trials):
{thought_block}

AVAILABLE OPERATORS AND DATA FIELDS:
{scoped_text}

ALLOWED DATA GROUPS: {', '.join(sorted(self.allowed_data_groups))}

EXPRESSION SYNTAX:
- rank(ts_delta(divide(close, volume), 120))
- rank(ts_regression(revenue, assets, 220, lag=110, rettype=2))
- rank(ts_zscore(close / enterprise_value, 120)) * rank(ts_rank(returns, 60))
- group_neutralize(rank(delta(close, 5)), subindustry)
- Arithmetic: +, -, *, / between expressions
- Unary minus: -rank(close)
- Keyword args: ts_regression(y, x, d, lag=N, rettype=N)

GUIDELINES:
- Use 2-3 rank(...) terms multiplied together — more orthogonal components = better
- Each component should capture a different signal (momentum, value, quality, stability)
- Always wrap time-series operators in rank() for cross-sectional comparability
- Use divide() for ratios, not raw division with /
- Group operators: group_rank(expr, subindustry), group_neutralize(expr, sector)
- Start with decay=0; only increase if turnover is too high
- DO NOT repeat expressions from the history above

RESPONSE FORMAT (exactly):
ALPHA: [single-line alpha expression]
REASONING: [1-2 sentence explanation of the economic signal]
PARAMETERS_JSON: {{"decay": 0, "delay": 1, "neutralization": "subindustry", "truncation": 0.1}}
""".strip()
        return prompt

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def generate_alpha(
        self,
        strategy: str = "",
        history: List[Dict] | None = None,
        trial_index: int | None = None,
    ) -> Dict[str, Any]:
        """Generate a single alpha using the LLM."""
        history = history or []
        prompt = self._build_prompt(strategy, history)

        # Log prompt
        try:
            os.makedirs('logs', exist_ok=True)
            with open('logs/agent_prompts.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n{datetime.now().isoformat()}\n{prompt}\n")
        except Exception:
            pass

        # Call LLM
        try:
            response_text = await self.llm.generate(prompt)
        except Exception as e:
            return {'success': False, 'error': f"LLM call failed: {e}"}

        # Extract components
        expression = _extract_alpha_code(response_text)
        reasoning = _extract_reasoning(response_text)
        params = _extract_parameters(response_text)

        if not expression:
            return {
                'success': False,
                'error': 'Failed to extract expression from LLM response',
                'raw_response': response_text,
            }

        if not _validate_syntax(expression):
            return {
                'success': False,
                'error': f'Invalid syntax: {expression}',
                'raw_response': response_text,
            }

        # Check for exact duplicates in recent history
        recent_codes = {h['code'] for h in history[-10:]}
        if expression in recent_codes:
            return {
                'success': False,
                'error': 'Duplicate expression',
                'expression': expression,
            }

        return {
            'success': True,
            'expression': expression,
            'reasoning': reasoning,
            'parameters': params,
            'raw_response': response_text,
            'trial_index': trial_index,
        }

    async def run(
        self,
        n_trials: int = 60,
        strategy: str = "",
        run_id: int | None = None,
        on_result: Callable[[int, EvaluationResult], None] | None = None,
    ) -> List[EvaluationResult]:
        """
        Run the full research loop.

        Args:
            n_trials: Number of alpha generation attempts
            strategy: Strategy description for the LLM
            run_id: Optional run_id to group results in the database
            on_result: Optional callback(trial_index, result) for each trial
        """
        # Create run in database
        if run_id is None and self.evaluator.db:
            run_id = self.evaluator.db.create_run(
                strategy=strategy,
                llm_model=self.llm.model_name,
                config={
                    'n_trials': n_trials,
                    'allowed_groups': sorted(self.allowed_data_groups),
                    'data_fields_pct': self.data_fields_pct,
                    'operators_pct': self.operators_pct,
                },
            )

        history = self.evaluator.get_history(limit=self.history_lookback)
        results: List[EvaluationResult] = []
        successful = 0
        failed = 0

        print(f"\n🚀 Alpha Research Agent — {n_trials} trials")
        print(f"   LLM: {self.llm.model_name}")
        print(f"   Data groups: {', '.join(sorted(self.allowed_data_groups))}")
        print(f"   Strategy: {strategy or '(general)'}")
        print("=" * 60)

        for trial in range(1, n_trials + 1):
            print(f"\n--- Trial {trial}/{n_trials} ---")

            # Generate
            gen_result = await self.generate_alpha(
                strategy=strategy, history=history, trial_index=trial
            )

            if not gen_result['success']:
                failed += 1
                print(f"  ✗ Generation failed: {gen_result.get('error', '?')}")
                self.thought_history.append({
                    'trial': trial,
                    'expression': gen_result.get('expression', ''),
                    'outcome': {'success': False, 'error': gen_result.get('error', '')},
                })
                continue

            expression = gen_result['expression']
            params = gen_result.get('parameters', {})
            reasoning = gen_result.get('reasoning', '')
            print(f"  Expression: {expression[:80]}...")
            print(f"  Reasoning: {reasoning[:60]}...")

            # Evaluate
            eval_result = self.evaluator.evaluate(
                expression=expression,
                params=params,
                run_id=run_id,
                trial_index=trial,
                reasoning=reasoning,
            )
            results.append(eval_result)

            if eval_result.success:
                successful += 1
                print(
                    f"  ✓ Sharpe={eval_result.sharpe:.3f} "
                    f"Fitness={eval_result.fitness:.3f} "
                    f"Turnover={eval_result.turnover:.3f} "
                    f"Checks={eval_result.passed_checks}"
                )

                # Add to history
                history.append({
                    'alpha_id': f'trial_{trial}',
                    'fitness': eval_result.fitness or 0,
                    'sharpe': eval_result.sharpe or 0,
                    'code': expression,
                })
                if len(history) > self.history_lookback:
                    history = history[-self.history_lookback:]

                self.thought_history.append({
                    'trial': trial,
                    'expression': expression,
                    'outcome': {
                        'success': True,
                        'sharpe': eval_result.sharpe,
                        'fitness': eval_result.fitness,
                    },
                })
            else:
                failed += 1
                print(f"  ✗ Evaluation failed: {eval_result.error}")
                self.thought_history.append({
                    'trial': trial,
                    'expression': expression,
                    'outcome': {'success': False, 'error': eval_result.error or ''},
                })

            if on_result:
                on_result(trial, eval_result)

            # Progress
            if trial % 10 == 0:
                print(f"\n📊 Progress: {trial}/{n_trials} | ✓{successful} ✗{failed}")

            await asyncio.sleep(0.1)  # small delay between trials

        # Finish run
        if self.evaluator.db and run_id:
            self.evaluator.db.finish_run(run_id)

        # Summary
        print(f"\n{'='*60}")
        print(f"🎯 RESULTS: {n_trials} trials | ✓{successful} ✗{failed}")

        successful_results = [r for r in results if r.success]
        if successful_results:
            best = max(successful_results, key=lambda r: r.fitness or 0)
            print(f"\n🏆 Best alpha:")
            print(f"   Expression: {best.expression}")
            print(f"   Sharpe: {best.sharpe:.3f}")
            print(f"   Fitness: {best.fitness:.3f}")
            print(f"   Turnover: {best.turnover:.3f}")

            passing = [r for r in successful_results if r.passed_checks >= 5]
            if passing:
                print(f"\n✅ {len(passing)} alphas passed quality checks")

        return results

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset in-run state for a new campaign."""
        self.thought_history.clear()
        self._scoped_text = None
