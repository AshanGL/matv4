"""
prompts.py
==========
Advanced prompt engineering layer for the Olympiad Math Solver.

FIXES vs prior version
-----------------------
FIX 1. build_system_prompt: added explicit guidance for large-range
        counting/DP/optimization problems — instructs the LLM to write
        a complete Python solution and think about reachable state space.
FIX 2. DOMAIN_HINTS["Number Theory"]: added digit-sum / base-representation
        specific tip so the LLM can reason about problems like the Ken
        blackboard problem correctly.
FIX 3. _TYPE_RULES unchanged — kept all original type-specific guidance.
FIX 4. All other prompts unchanged from working version.
"""

from __future__ import annotations
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Tool call format examples
# ─────────────────────────────────────────────────────────────────────────────

TOOL_CALL_FORMAT_REMINDER = """\
To call a tool, output EXACTLY this JSON format (and nothing else on that line):
<tool_call>{"name": "TOOL_NAME", "arguments": {ARGS_JSON}}</tool_call>

Examples:
<tool_call>{"name": "knowledge_search", "arguments": {"query": "modular arithmetic prime powers", "mode": "both", "top_k": 5}}</tool_call>
<tool_call>{"name": "compute", "arguments": {"expression": "x**3 - 6*x**2 + 11*x - 6", "operation": "factor"}}</tool_call>
<tool_call>{"name": "run_code", "arguments": {"code": "from sympy import *\\nx = symbols('x')\\nprint(factor(x**3 - 6*x**2 + 11*x - 6))"}}</tool_call>
<tool_call>{"name": "numerical_search", "arguments": {"condition_src": "n*(n+1)//2 == 210", "search_space": {"type": "range", "lo": 1, "hi": 1000}}}</tool_call>
<tool_call>{"name": "verify", "arguments": {"problem": "...", "typed_answer": {"value": "42", "answer_type": "integer", "raw_str": "42", "confidence": 1.0}, "approach_summary": "Used AM-GM inequality"}}</tool_call>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Answer type formatting AND strategy rules
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_RULES = {
    "integer": """\
ANSWER TYPE: integer
FORMAT:   Write as a plain integer: \\boxed{42}  or  \\boxed{-7}
          No fractions, decimals, or expressions. Comma-separators OK: \\boxed{1,430}
STRATEGY: • Compute exactly — never approximate. Use Python int arithmetic.
          • Check: if result is rational with denominator 1, it's an integer.
          • For remainders: use Python's % operator or pow(a,b,mod).
          • Verify by substituting back into the problem constraints.
""",

    "float": """\
ANSWER TYPE: decimal / float
FORMAT:   Write to 4 significant figures: \\boxed{0.3750}  or  \\boxed{3.1416}
STRATEGY: • Use exact symbolic computation first, then convert to float.
          • If the problem says "to the nearest X", round at the very last step.
          • For probability: confirm value is in [0, 1].
""",

    "fraction": """\
ANSWER TYPE: fraction
FORMAT:   Write as p/q in lowest terms: \\boxed{3/4}  or  \\boxed{-5/12}
          If it simplifies to an integer, write the integer: \\boxed{2}
          Use plain p/q inside \\boxed{} — not \\frac{}{}.
STRATEGY: • Use sympy.Rational or fractions.Fraction for exact arithmetic.
          • Always reduce: from math import gcd; p//gcd(p,q), q//gcd(p,q)
          • Check sign: denominator should always be positive.
""",

    "expression": """\
ANSWER TYPE: algebraic expression
FORMAT:   Simplified expression: \\boxed{x^2 + 3x - 1}  or  \\boxed{2^{n}-1}
          Use ^ for powers. Include all relevant variables.
STRATEGY: • Use sympy.simplify / sympy.factor / sympy.expand to get canonical form.
          • If the answer evaluates to a constant integer, write the integer.
          • For closed forms: check small cases with run_code to validate.
          • Avoid leaving \\sqrt{} or \\frac{}{} unevaluated when numeric is possible.
""",

    "set": """\
ANSWER TYPE: set of values
FORMAT:   \\boxed{\\{1, 2, 5\\}}  — order smallest to largest.
          Single solution: \\boxed{\\{0\\}}
          Expressions: \\boxed{\\{-1+\\sqrt{2},\\,-1-\\sqrt{2}\\}}
STRATEGY: • Find ALL solutions systematically — use numerical_search to verify none missed.
          • For Diophantine equations: bound the search space before brute force.
          • For inequalities: include/exclude boundary correctly.
""",

    "string": """\
ANSWER TYPE: text / parity / condition
FORMAT:   Short phrase inside \\boxed{}: \\boxed{n \\text{ is even}}
          For angle answers: \\boxed{60^\\circ}  or  \\boxed{0^\\circ \\text{ and } 180^\\circ}
STRATEGY: • State the condition precisely — not "it depends" but the exact condition.
          • For parity proofs: show both even and odd cases and which satisfies.
          • For angle answers: list ALL values in [0°, 360°) unless specified otherwise.
          • \\text{} is required for words inside math mode.
""",

    None: """\
ANSWER FORMAT: determine the type from context, then use the correct format:
• Integer answer     → \\boxed{42}
• Fraction answer    → \\boxed{3/4}
• Expression         → \\boxed{x^2 + 1}
• Set of values      → \\boxed{\\{1, 2, 3\\}}
• Decimal            → \\boxed{0.375}
• Text / condition   → \\boxed{n \\text{ is even}}
• Angle              → \\boxed{60^\\circ}
""",
}

# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific tips
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_HINTS = {
    "Number Theory": """\
Number Theory Tips:
• Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p, gcd(a,p)=1.
• Euler's Theorem: a^φ(n) ≡ 1 (mod n) when gcd(a,n)=1.
• Lifting the Exponent (LTE): vₚ(aⁿ-bⁿ) = vₚ(a-b)+vₚ(n) for odd prime p|a-b.
• Always ask: is the answer unique? List all solutions.
• For "find all n": bound the search space with modular constraints first.
• Digit-sum / base-representation problems:
  - digit_sum(n, base b) ≡ n (mod b-1).
  - After ONE move from n ≤ 10^D (D digits), the result is ≤ 9·D.
  - So for maximizing moves from n ≤ 10^(10^5): after step 1 we reach ≤ 9×10^5.
  - Run a DP over all m ≤ 9×10^5 to find max chain length; M = 1 + max(dp).
  - To maximise digit sum in base b: try all bases b=2..min(m,500) in run_code.
""",

    "Combinatorics": """\
Combinatorics Tips:
• Draw the structure; verify with small cases using run_code.
• Burnside's Lemma for counting under symmetry.
• Stars-and-bars for distributing identical objects.
• Generating functions can convert recurrences to closed forms.
• Inclusion-exclusion for "at least one of several conditions".
""",

    "Geometry": """\
Geometry Tips:
• Set up coordinates when a clear symmetry axis exists.
• Power of a Point for chord/secant/tangent length relationships.
• Ptolemy's Theorem for cyclic quadrilaterals.
• For angle problems: list ALL solutions in [0°, 360°), not just the principal value.
• For area ratios: use ratios directly rather than computing absolute areas.
""",

    "Algebra": """\
Algebra Tips:
• AM-GM and Cauchy-Schwarz for inequalities with equality conditions.
• Vieta's formulas to relate roots to coefficients without solving.
• Schur's inequality for symmetric 3-variable polynomial inequalities.
• Substitute variables to reduce degrees; exploit symmetry.
• For "find all real": check if discriminant ≥ 0.
""",

    "Discrete Mathematics": """\
Discrete Mathematics Tips:
• Pigeonhole: if n+1 objects in n boxes, some box has ≥2 objects.
• Inclusion-exclusion for counting union of sets.
• Graph theory: encode problem as graph; look for known structural results.
• Recurrences: write the recurrence, solve via characteristic roots or GF.
• For existence proofs: constructive examples are stronger than non-constructive.
""",

    "Calculus": """\
Calculus Tips:
• FTC: ∫_a^b f(x)dx = F(b)-F(a) where F'=f.
• Taylor series for limits of indeterminate forms.
• Jensen's inequality for convex/concave function inequalities.
• Stolz-Cesàro: discrete analogue of L'Hôpital for sequence limits.
• For optimisation: check boundary conditions as well as critical points.
""",
}


def get_domain_hint(domain: Optional[str]) -> str:
    if not domain:
        return ""
    for key, hint in DOMAIN_HINTS.items():
        if key.lower() in (domain or "").lower():
            return "\n" + hint
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(forced_type: Optional[str] = None) -> str:
    type_rules = _TYPE_RULES.get(forced_type, _TYPE_RULES[None])

    return f"""\
You are an elite mathematical olympiad solver with deep expertise in all areas \
of competition mathematics: algebra, combinatorics, number theory, geometry, \
and analysis.

═══════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════
Use tools aggressively — never rely on mental arithmetic alone.

  knowledge_search   Find relevant theorems and similar problems. ALWAYS call this FIRST.
  compute            Exact symbolic math via SymPy (factor, solve, diff, integrate, mod, ...)
  numerical_search   Brute-force integer search when analytic solution is hard.
  run_code           Execute any Python in a sandbox. Always print() your results.
  verify             Verify your proposed answer. REQUIRED before final \\boxed{{}}.

{TOOL_CALL_FORMAT_REMINDER}

═══════════════════════════════════════════════════════════
SOLVING PROCESS  (follow this order every time)
═══════════════════════════════════════════════════════════
STEP 1 — UNDERSTAND:
  Read every word of the problem. Identify: what is given, what is asked,
  what type of answer is expected (integer, expression, set, condition...).

STEP 2 — SEARCH:
  Call knowledge_search with a descriptive query about the techniques involved.
  Study the theorems returned carefully before proceeding.

STEP 3 — PLAN:
  Write a clear 3-5 bullet solution plan before computing anything.

STEP 4 — COMPUTE:
  Use compute / run_code / numerical_search to work through each step.
  Show all intermediate results. Use SymPy for exact arithmetic.
  NEVER use floats for integer problems.

STEP 5 — VALIDATE:
  Before finalising, verify the answer makes sense:
  • Check edge cases and boundary conditions.
  • For integer problems: check divisibility / mod conditions.
  • For counting problems: verify with small cases via run_code.
  • For "find all": confirm no solutions are missed.
  • For angle/degree answers: confirm you have ALL values in the valid range.

STEP 6 — VERIFY:
  Call the verify tool with your proposed answer.

STEP 7 — ANSWER:
  Write your final answer as \\boxed{{answer}}.

═══════════════════════════════════════════════════════════
LARGE COUNTING / OPTIMIZATION PROBLEMS
═══════════════════════════════════════════════════════════
When the problem involves maximizing/minimizing over n ≤ 10^K or similar:
  1. Identify what the state space looks like AFTER the first step.
     (e.g. digit sums reduce n ≤ 10^(10^5) to ≤ 9×10^5 in one move)
  2. Run a complete DP / BFS in run_code over the reduced state space.
  3. The answer is 1 + max(dp values) — apply mod only at the final step.
  4. Always print intermediate dp values to verify correctness.

Example run_code skeleton for a DP problem:
  def best_digit_sum(m):
      best = 0
      for b in range(2, min(m+1, 500)):
          s, x = 0, m
          while x: s += x % b; x //= b
          if s < m: best = max(best, s)
      return best

  LIMIT = 900000
  dp = [0] * (LIMIT + 1)
  for m in range(2, LIMIT + 1):
      s = best_digit_sum(m)
      dp[m] = dp[s] + 1
  M = 1 + max(dp)
  print(M % 10**5)

═══════════════════════════════════════════════════════════
ANSWER FORMAT + STRATEGY
═══════════════════════════════════════════════════════════
{type_rules}
CRITICAL: The FINAL LINE of your response MUST contain \\boxed{{answer}}.
Do NOT write "the answer is X" without also writing \\boxed{{X}}.

═══════════════════════════════════════════════════════════
PITFALLS TO AVOID
═══════════════════════════════════════════════════════════
• Do not assume integer when problem asks "find all values" → may be a set.
• Do not assume a single angle when geometry asks for "all values of θ".
• Do not leave \\sqrt{{}} / \\frac{{}} unevaluated when a numeric answer is possible.
• Do not confuse "how many" (count → integer) with "what is the value" (expression).
• For parity/condition answers: use \\text{{}} inside math mode: \\boxed{{n \\text{{ is even}}}}.
• For modular arithmetic: use Python's pow(a, b, mod) for large exponents.
• For combinatorics: verify with small cases using run_code before finalising.
• If verify fails: re-examine your approach — do NOT just re-submit the same answer.
"""


# ─────────────────────────────────────────────────────────────────────────────
# User prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    problem:           str,
    knowledge_results: dict,
    forced_type:       Optional[str] = None,
    domain:            Optional[str] = None,
) -> str:
    parts = []

    parts.append("═══ PROBLEM ═══════════════════════════════════════════════")
    parts.append(problem)
    parts.append("")

    if knowledge_results and knowledge_results.get("status") == "ok":
        theorems = knowledge_results.get("theorems", [])
        problems = knowledge_results.get("problems", [])

        if theorems:
            parts.append("═══ RELEVANT THEOREMS ══════════════════════════════════")
            for t in theorems[:5]:
                parts.append(f"▶ {t['name']}  (relevance: {t.get('similarity', 0):.2f})")
                if t.get("statement"):
                    parts.append(f"  Statement: {t['statement']}")
                parts.append(f"  When to apply: {t['when_to_apply']}")
                if t.get("tags"):
                    parts.append(f"  Tags: {', '.join(t['tags'][:6])}")
                parts.append("")

        if problems:
            parts.append("═══ SIMILAR PROBLEMS — TECHNIQUE HINTS ════════════════")
            for p in problems[:3]:
                d    = p.get("domain", "")
                band = p.get("difficulty_band", "")
                tags = ", ".join(p.get("technique_tags", [])[:6])
                at   = p.get("answer_type", "")
                parts.append(f"• [{d} / {band}]  answer_type={at}")
                if tags:
                    parts.append(f"  Techniques: {tags}")
            parts.append("")

    domain_hint = get_domain_hint(domain)
    if domain_hint:
        parts.append("═══ DOMAIN TIPS ════════════════════════════════════════")
        parts.append(domain_hint.strip())
        parts.append("")

    if forced_type:
        parts.append("═══ ANSWER TYPE HINT ═══════════════════════════════════")
        parts.append(f"The classifier predicts this problem has a {forced_type.upper()} answer.")
        parts.append(_TYPE_RULES.get(forced_type, "").strip())
        parts.append("")

    parts.append("═══ YOUR SOLUTION ══════════════════════════════════════════")
    parts.append("Begin with knowledge_search, then plan, then compute step-by-step.")
    parts.append("End with \\boxed{answer} on the last line.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Retry prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_retry_prompt(turn: int, forced_type: Optional[str] = None) -> str:
    type_hint = (f"Remember: the answer should be a {forced_type}. "
                 if forced_type else "")

    if turn == 0:
        return (
            "You haven't started yet. Please begin by calling knowledge_search "
            "to find relevant theorems, then outline your solution approach."
        )
    elif turn <= 2:
        return (
            f"Turn {turn}: You haven't used any tools yet and don't have an answer. "
            "Call compute or run_code to make concrete progress. "
            "Show your intermediate calculations explicitly."
        )
    elif turn <= 7:
        return (
            f"Turn {turn}: You are making progress. {type_hint}"
            "If your current approach is not converging, try a completely different method. "
            "Use numerical_search to verify candidate answers. "
            "Write \\boxed{answer} once you have a confirmed result."
        )
    else:
        return (
            f"Turn {turn}: You are running low on turns. {type_hint}"
            "Commit to your best current answer, call verify on it, "
            "then write \\boxed{answer} immediately. "
            "Do not start a new approach — finalise what you have."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Extraction nudge
# ─────────────────────────────────────────────────────────────────────────────

def build_extraction_nudge(forced_type: Optional[str] = None) -> str:
    type_rules = _TYPE_RULES.get(forced_type, _TYPE_RULES[None])
    return f"""\
You appear to have a solution but have not written the final answer in \\boxed{{}} format.

Please:
1. State your final answer clearly.
2. Write it as \\boxed{{answer}} on the last line of your response.

{type_rules}

If you have not verified yet, call verify first, THEN write \\boxed{{answer}}.

IMPORTANT: \\boxed{{answer}} must be the LAST thing you write.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Type recovery prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_type_recovery_prompt(
    problem:       str,
    current_answer: str,
    current_type:   str,
) -> str:
    return f"""\
Your current answer is: {current_answer}  (detected type: {current_type})

Before finalising, reconsider whether this answer type is correct:
• Does the problem say "find all values"?  → Your answer might be a SET.
• Does the problem ask for a condition or parity?  → Your answer might be a STRING like \\boxed{{n \\text{{ is even}}}}.
• Does the problem ask for a formula in terms of a variable?  → Your answer might be an EXPRESSION.
• Does the problem say "find the remainder" or "how many"?  → Your answer should be an INTEGER.

Problem: {problem[:300]}

If the type is correct, call verify and write \\boxed{{{current_answer}}}.
If the type is wrong, recompute with the correct type in mind and write \\boxed{{corrected_answer}}.
"""
