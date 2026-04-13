"""
prompts.py
==========
Advanced prompt engineering layer for the Olympiad Math Solver.

Design principles
-----------------
1. Type-aware system prompt: tells the LLM EXACTLY what format is expected
   for each answer type (integer, fraction, expression, set, string).
2. Chain-of-thought structure: forces explicit plan → compute → verify flow.
3. Tool-call format is shown as a concrete example, not abstract description.
4. Knowledge search results are injected with full theorem statements.
5. Retry and nudge prompts are calibrated to the type and the turn number.
6. Low-temperature extraction prompt extracts answers even from verbose output.
"""

from __future__ import annotations
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Tool call format examples (shown inline to the LLM)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_CALL_FORMAT_REMINDER = """\
To call a tool, output EXACTLY this JSON format (nothing else on that line):
<tool_call>{"name": "TOOL_NAME", "arguments": {ARGS_JSON}}</tool_call>

Examples:
<tool_call>{"name": "knowledge_search", "arguments": {"query": "modular arithmetic prime powers", "mode": "both", "top_k": 5}}</tool_call>
<tool_call>{"name": "compute", "arguments": {"expression": "x**3 - 6*x**2 + 11*x - 6", "operation": "factor"}}</tool_call>
<tool_call>{"name": "run_code", "arguments": {"code": "from sympy import *\\nx = symbols('x')\\nprint(factor(x**3 - 6*x**2 + 11*x - 6))"}}</tool_call>
<tool_call>{"name": "numerical_search", "arguments": {"condition_src": "n*(n+1)//2 == 210", "search_space": {"type": "range", "lo": 1, "hi": 1000}}}</tool_call>
<tool_call>{"name": "verify", "arguments": {"problem": "...", "typed_answer": {"value": "42", "answer_type": "integer", "raw_str": "42", "confidence": 1.0}, "approach_summary": "Used AM-GM inequality"}}</tool_call>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Answer type formatting rules
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_RULES = {
    "integer": """\
ANSWER TYPE: integer
• Write the answer as a plain integer: \\boxed{42}
• Do NOT write fractions, decimals, or expressions.
• If the answer is negative, include the sign: \\boxed{-7}
• Comma-separated thousands are OK in the box: \\boxed{1,430}
""",

    "float": """\
ANSWER TYPE: decimal / float
• Write to 4 significant figures unless the problem specifies otherwise.
• Example: \\boxed{0.3750}  or  \\boxed{3.1416}
• Use exact fractions if possible, then convert: 3/8 → 0.3750
""",

    "fraction": """\
ANSWER TYPE: fraction
• Write as p/q in lowest terms: \\boxed{3/4}
• If the fraction is negative: \\boxed{-5/12}
• Use \\frac{}{} only inside LaTeX; in \\boxed{} use plain p/q.
• If the answer simplifies to an integer, write the integer: \\boxed{2}
""",

    "expression": """\
ANSWER TYPE: algebraic expression
• Write a simplified expression using standard notation: \\boxed{x^2 + 3x - 1}
• Use ^ for powers, * for multiplication if needed.
• Simplify fully using SymPy before writing the final expression.
• Include all variables mentioned in the problem.
• If the answer is a specific value (integer/fraction), write that value.
Examples: \\boxed{n(n-1)/2}  \\boxed{2^{n}-1}  \\boxed{\\sqrt{3}/2}
""",

    "set": """\
ANSWER TYPE: set of values
• List all solutions inside braces: \\boxed{{1, 2, 5}}
• Order elements from smallest to largest.
• If the answer involves expressions: \\boxed{{-1 + \\sqrt{2}, -1 - \\sqrt{2}}}
• For a single solution, still use set notation: \\boxed{{0}}
""",

    "string": """\
ANSWER TYPE: text / parity / condition answer
• Write the answer as a short phrase or word: \\boxed{n \\text{ is even}}
• Do not over-explain — just the answer inside \\boxed{}.
""",

    None: """\
ANSWER FORMAT: determine the type from context, then:
• Integer answer → \\boxed{42}
• Fraction answer → \\boxed{3/4}
• Expression → \\boxed{x^2 + 1}
• Set of values → \\boxed{{1, 2, 3}}
• Decimal → \\boxed{0.375}
• Text condition → \\boxed{n \\text{ is even}}
""",
}

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
You have five tools. Use them aggressively — do not rely on mental arithmetic.

  knowledge_search   Search for relevant theorems and similar problems. ALWAYS call this FIRST.
  compute            Exact symbolic math via SymPy (factor, solve, diff, integrate, modular, ...)
  numerical_search   Brute-force integer search when analytic solution is hard.
  run_code           Execute any Python in a sandbox. Always print() your results.
  verify             Verify your proposed answer. REQUIRED before final \\boxed{{}}.

{TOOL_CALL_FORMAT_REMINDER}

═══════════════════════════════════════════════════════════
SOLVING PROCESS  (follow this order)
═══════════════════════════════════════════════════════════
STEP 1 — UNDERSTAND:  Read the problem carefully. Identify: what is given, what \
is asked, what constraints exist.

STEP 2 — SEARCH:  Call knowledge_search with a descriptive query about the \
mathematical techniques involved. Study the theorems returned.

STEP 3 — PLAN:  Write a clear solution plan in 3-5 bullet points before computing.

STEP 4 — COMPUTE:  Use compute / run_code / numerical_search to work through \
the solution. Show all intermediate steps. Use SymPy for exact arithmetic — \
NEVER use floats for integer problems.

STEP 5 — CHECK:  Before finalising, verify your answer makes sense:
  • Check edge cases and boundary conditions.
  • If it's an integer problem, check divisibility / mod conditions.
  • If it's a counting problem, check small cases by hand.

STEP 6 — VERIFY:  Call the verify tool with your proposed answer.

STEP 7 — ANSWER:  Write your final answer as \\boxed{{answer}}.

═══════════════════════════════════════════════════════════
ANSWER FORMAT
═══════════════════════════════════════════════════════════
{type_rules}
CRITICAL: The final line of your response MUST contain \\boxed{{answer}}.
Do not write "the answer is X" without also writing \\boxed{{X}}.

═══════════════════════════════════════════════════════════
COMMON PITFALLS TO AVOID
═══════════════════════════════════════════════════════════
• Do not assume the answer is an integer when the problem says "find all values".
• Do not leave \\sqrt{{}} or \\frac{{}} unevaluated when a numeric answer is possible.
• Do not confuse "how many" (count → integer) with "what is the sum" (may be expression).
• For modular arithmetic, use Python's pow(a, b, mod) for large exponents.
• For geometry, always set up coordinates or use trigonometric identities.
• For combinatorics, verify with small cases using run_code.
"""


# ─────────────────────────────────────────────────────────────────────────────
# User prompt (with knowledge search results injected)
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    problem: str,
    knowledge_results: dict,
    forced_type: Optional[str] = None,
) -> str:
    parts = []

    # Problem statement
    parts.append("═══ PROBLEM ═══════════════════════════════════════════════")
    parts.append(problem)
    parts.append("")

    # Knowledge search results
    if knowledge_results and knowledge_results.get("status") == "ok":
        theorems = knowledge_results.get("theorems", [])
        problems = knowledge_results.get("problems", [])

        if theorems:
            parts.append("═══ RELEVANT THEOREMS (from knowledge search) ══════════")
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
                domain = p.get("domain", "")
                band   = p.get("difficulty_band", "")
                tags   = ", ".join(p.get("technique_tags", [])[:6])
                atype  = p.get("answer_type", "")
                parts.append(f"• [{domain} / {band}]  answer_type={atype}")
                if tags:
                    parts.append(f"  Techniques: {tags}")
            parts.append("")

    # Type hint
    if forced_type:
        parts.append(f"═══ ANSWER TYPE HINT ═══════════════════════════════════")
        parts.append(f"The classifier predicts this problem has a {forced_type.upper()} answer.")
        parts.append(_TYPE_RULES.get(forced_type, ""))

    parts.append("═══ YOUR SOLUTION ══════════════════════════════════════════")
    parts.append("Begin with knowledge_search, then plan, then compute step-by-step.")
    parts.append("End with \\boxed{answer}.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Retry prompt (when LLM makes no progress)
# ─────────────────────────────────────────────────────────────────────────────

def build_retry_prompt(turn: int, forced_type: Optional[str] = None) -> str:
    if turn == 0:
        return (
            "You haven't started yet. Please begin by calling knowledge_search "
            "to find relevant theorems, then outline your solution approach."
        )
    elif turn <= 3:
        return (
            f"Turn {turn}: You haven't used any tools yet and don't have an answer. "
            "Please call compute or run_code to make progress. "
            "Show your intermediate calculations."
        )
    else:
        type_hint = (
            f"Remember: the answer should be a {forced_type}. "
            if forced_type else ""
        )
        return (
            f"Turn {turn}: You are making progress but haven't reached a conclusion. "
            f"{type_hint}"
            "Please verify your intermediate results and work toward a final answer. "
            "Call verify when you have a candidate answer, then write \\boxed{answer}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Extraction nudge (when LLM solved the problem but forgot \boxed{})
# ─────────────────────────────────────────────────────────────────────────────

def build_extraction_nudge(forced_type: Optional[str] = None) -> str:
    type_rules = _TYPE_RULES.get(forced_type, _TYPE_RULES[None])
    return f"""\
You appear to have a solution but haven't written the final answer in \\boxed{{}} format.

Please:
1. State your final answer clearly.
2. Write it as \\boxed{{answer}} on the last line.

{type_rules}

If you haven't verified yet, call verify first, then write \\boxed{{answer}}.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific prompt enhancers (injected into user prompt for hard problems)
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_HINTS = {
    "Number Theory": """\
Number Theory Tips:
• Use Fermat's Little Theorem for prime moduli: a^(p-1) ≡ 1 (mod p)
• Use Euler's theorem for composite moduli: a^φ(n) ≡ 1 (mod n) when gcd(a,n)=1
• For divisibility, try the Lifting-the-Exponent lemma
• Always check: is the answer unique? Are there multiple solutions to enumerate?
""",

    "Combinatorics": """\
Combinatorics Tips:
• Draw the structure before computing. Try small cases first.
• Consider symmetry and Burnside's lemma for counting under equivalence.
• Generating functions can turn recurrences into closed forms.
• Stars-and-bars for distributing identical objects; multinomials for distinct.
""",

    "Geometry": """\
Geometry Tips:
• Set up coordinates when the problem has a clear symmetry axis.
• Use trigonometric form for circle problems.
• Power of a Point for chord/secant/tangent relationships.
• For area problems, consider the ratio approach before direct computation.
""",

    "Algebra": """\
Algebra Tips:
• Substitute variables to reduce degrees; look for symmetry.
• AM-GM and Cauchy-Schwarz for inequalities with equality conditions.
• Vieta's formulas to relate polynomial roots to coefficients without solving.
• Schur's inequality for symmetric 3-variable polynomial inequalities.
""",

    "Discrete Mathematics": """\
Discrete Mathematics / Combinatorics Tips:
• Pigeonhole: if impossible to avoid, a collision must exist.
• Inclusion-exclusion for "at least one of" counting.
• Graph theory: encode the problem as a graph and look for known results.
• Recurrences: write the recurrence, then solve with characteristic roots or GF.
""",
}


def get_domain_hint(domain: Optional[str]) -> str:
    if domain is None:
        return ""
    for key in DOMAIN_HINTS:
        if key.lower() in (domain or "").lower():
            return "\n" + DOMAIN_HINTS[key]
    return ""
