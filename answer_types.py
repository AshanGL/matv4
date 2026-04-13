"""
answer_types.py
===============
Multi-type answer system for the Olympiad Math Solver.

Supports: integer, float, fraction, expression, set, string
Every component downstream (verify, vote, compare) works with
TypedAnswer objects — never raw strings or plain ints.

IMPROVEMENTS vs prior version
-------------------------------
1. extract_integer: strips trailing punctuation so "-54." → -54.
2. AnswerTypeDetector.detect: expression+plain-integer box → downgrade to integer.
3. extract_raw_boxed: rejects single-char garbage matches from fallback patterns.
4. _infer_from_content: degree-symbol answers (° / \\circ) routed to string.
5. extract_raw_boxed: handles nested braces in \\boxed{\\frac{a}{b}} correctly.
6. answers_match: cross-type integer vs expression comparison added.
7. AnswerTypeDetector: parity/condition signals ("is even","is odd","for all n").
8. extract_string: normalises \\text{} and \\mathrm{} LaTeX macros.
9. select_best_answer: confidence-weighted group score for tiebreaks.
10. extract_expression: added \\binom normalisation.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Core dataclass
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_TYPES = ("integer", "float", "fraction", "expression", "set", "string")


@dataclass
class TypedAnswer:
    """
    Canonical container for any answer produced by the solver.

    Attributes
    ----------
    value        : int | float | sp.Expr | sp.Rational | frozenset | str
    answer_type  : one of ANSWER_TYPES
    raw_str      : original string from \\boxed{} — preserved for debugging
    confidence   : 0.0–1.0  extraction reliability
    tolerance    : absolute tolerance used for float/fraction comparison
    attempt_idx  : which solver attempt produced this (for voting)
    """
    value:       Any
    answer_type: str
    raw_str:     str
    confidence:  float        = 1.0
    tolerance:   float        = 1e-9
    attempt_idx: int          = -1
    extra:       dict         = field(default_factory=dict)

    def __post_init__(self):
        if self.answer_type not in ANSWER_TYPES:
            raise ValueError(f"Unknown answer_type: {self.answer_type!r}. "
                             f"Must be one of {ANSWER_TYPES}")

    def __repr__(self):
        return (f"TypedAnswer(type={self.answer_type}, value={self.value!r}, "
                f"conf={self.confidence:.2f})")

    def is_numeric(self) -> bool:
        return self.answer_type in ("integer", "float", "fraction")

    def as_float(self) -> Optional[float]:
        try:
            return float(self.value)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Answer type detector
# ─────────────────────────────────────────────────────────────────────────────

class AnswerTypeDetector:
    """
    Infers the expected answer type from the problem text + boxed content.
    Conservative — when unsure returns "string" to preserve raw text.
    """

    _INTEGER_SIGNALS = [
        "how many", "find the number", "count the", "number of",
        "largest integer", "smallest integer", "smallest positive integer",
        "greatest integer", "least positive integer",
        "remainder when", "last two digits", "last digit",
        "number of ways", "in how many", "total number",
    ]
    _FLOAT_SIGNALS = [
        "probability", "expected value", "to 2 decimal", "to 3 decimal",
        "to the nearest", "average", "mean value", "approximate",
    ]
    _FRACTION_SIGNALS = [
        "as a fraction", "in the form p/q", "express as a ratio",
        "in lowest terms", "reduced fraction",
    ]
    _EXPRESSION_SIGNALS = [
        "in terms of", "express", "simplify", "factor", "find all real",
        "find f(", "closed form", "general term", "polynomial",
    ]
    _SET_SIGNALS = [
        "find all values", "list all", "all solutions", "all integers",
        "which values", "find all x", "find all n", "find all pairs",
    ]
    # IMPROVEMENT 7: parity / condition answer signals → string
    _STRING_SIGNALS = [
        "is even", "is odd", "always", "never", "for all n",
        "prove that", "determine whether", "which of the following",
        "parity", "true or false",
    ]

    def detect(self, problem: str, boxed_content: str) -> str:
        p = problem.lower()
        b = boxed_content.strip()

        # Hard signals from problem text (most specific first)
        if any(s in p for s in self._FRACTION_SIGNALS):    return "fraction"
        if any(s in p for s in self._INTEGER_SIGNALS):     return "integer"
        if any(s in p for s in self._FLOAT_SIGNALS):       return "float"
        if any(s in p for s in self._SET_SIGNALS):         return "set"
        if any(s in p for s in self._STRING_SIGNALS):      return "string"
        if any(s in p for s in self._EXPRESSION_SIGNALS):  return "expression"

        detected = self._infer_from_content(b)

        # IMPROVEMENT 2: expression label but box is a plain integer → downgrade
        if detected == "expression" and re.fullmatch(r"-?\s*[\d,]+", b):
            return "integer"

        return detected

    def _infer_from_content(self, b: str) -> str:
        b = b.strip()

        # IMPROVEMENT 4: degree-symbol answers → string
        if "°" in b or "\\circ" in b or "^\\circ" in b:
            return "string"

        # IMPROVEMENT 7: explicit text macros → string
        if "\\text{" in b or "\\mathrm{" in b:
            return "string"

        # Pure integer (signed, comma-separated thousands OK)
        if re.fullmatch(r"-?\s*[\d,]+", b):
            return "integer"

        # Decimal float
        if re.fullmatch(r"-?\d+\.\d+", b):
            return "float"

        # Fraction  a/b  or  -a/b
        if re.fullmatch(r"-?\d+\s*/\s*-?\d+", b):
            return "fraction"

        # LaTeX fraction \frac{a}{b}
        if re.search(r"\\frac\s*\{", b):
            return "fraction"

        # Set notation  {1, 2, 3}  or  \{1, 2\}
        if (b.startswith("{") and b.endswith("}")) or \
           (b.startswith("\\{") and b.endswith("\\}")):
            return "set"

        # Comma-separated list of numbers without braces → set
        if re.fullmatch(r"-?[\d\./]+(,\s*-?[\d\./]+)+", b):
            return "set"

        # Contains algebra variable or math operators → expression
        if re.search(r"[a-wyzA-WYZ\^\\]", b) or \
           any(tok in b for tok in ["sqrt", "\\sqrt", "log", "\\log",
                                     "sin", "cos", "tan", "pi", "\\pi",
                                     "infty", "\\infty"]):
            return "expression"

        return "string"


# ─────────────────────────────────────────────────────────────────────────────
# Boxed content extractor
# ─────────────────────────────────────────────────────────────────────────────

# IMPROVEMENT 5: nested-brace aware pattern first
_BOXED_PATTERNS = [
    r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}",    # handles \boxed{\frac{a}{b}}
    r"\\boxed\s*\{\s*([^{}]+)\s*\}",              # simple \boxed{...}
    r"the\s+answer\s+is\s*[:\s]*([^\n.]{2,})",    # "the answer is ..."  (≥2 chars)
    r"final\s+answer\s*[:\s=]*([^\n.]{2,})",      # "final answer: ..."
    r"=\s*\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", # = \boxed{...}
]


def extract_raw_boxed(text: str) -> Optional[str]:
    """
    Pull the raw string from inside \\boxed{} in the LLM output.
    Returns None if nothing found.

    IMPROVEMENT 3: rejects single-char garbage matches.
    IMPROVEMENT 5: handles nested braces like \\boxed{\\frac{3}{4}}.
    """
    # Restrict to assistant section
    for split_tok in ["assistant\n", "assistant\r", "assistant"]:
        if split_tok in text.lower():
            idx  = text.lower().rfind(split_tok)
            text = text[idx:]
            break

    for pat in _BOXED_PATTERNS:
        matches = re.findall(pat, text, re.IGNORECASE | re.DOTALL)
        if matches:
            result = matches[-1].strip()
            # IMPROVEMENT 3: reject degenerate single-char matches
            if len(result) >= 2:
                return result

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Type-specific extractors
# ─────────────────────────────────────────────────────────────────────────────

def _sympy():
    import sympy
    return sympy


def extract_integer(raw: str) -> TypedAnswer:
    # IMPROVEMENT 1: strip trailing punctuation before parsing
    clean = raw.replace(",", "").replace(" ", "").strip()
    clean = re.sub(r"[.\s;:]+$", "", clean)        # strip trailing . ; :
    clean = re.sub(r"^\{(.*)\}$", r"\1", clean)    # strip outer braces {-7}
    try:
        val = int(clean)
        return TypedAnswer(value=val, answer_type="integer",
                           raw_str=raw, confidence=1.0)
    except ValueError:
        sp = _sympy()
        try:
            val = int(sp.Integer(clean))
            return TypedAnswer(value=val, answer_type="integer",
                               raw_str=raw, confidence=0.85)
        except Exception:
            return TypedAnswer(value=raw, answer_type="string",
                               raw_str=raw, confidence=0.1,
                               extra={"parse_error": "integer"})


def extract_float(raw: str) -> TypedAnswer:
    clean = raw.strip().replace(",", "")
    try:
        val = float(clean)
        return TypedAnswer(value=val, answer_type="float",
                           raw_str=raw, confidence=0.95,
                           tolerance=1e-6)
    except ValueError:
        return TypedAnswer(value=raw, answer_type="string",
                           raw_str=raw, confidence=0.1,
                           extra={"parse_error": "float"})


def extract_fraction(raw: str) -> TypedAnswer:
    sp = _sympy()

    # a/b form
    m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", raw)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return TypedAnswer(value=raw, answer_type="string",
                               raw_str=raw, confidence=0.0,
                               extra={"parse_error": "division by zero"})
        val = sp.Rational(num, den)
        return TypedAnswer(value=val, answer_type="fraction",
                           raw_str=raw, confidence=0.98)

    # \frac{a}{b}
    m2 = re.search(r"\\frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}", raw)
    if m2:
        val = sp.Rational(int(m2.group(1)), int(m2.group(2)))
        return TypedAnswer(value=val, answer_type="fraction",
                           raw_str=raw, confidence=0.95)

    # Decimal that is actually rational e.g. "0.75"
    try:
        f = float(raw.strip())
        val = sp.Rational(f).limit_denominator(10000)
        return TypedAnswer(value=val, answer_type="fraction",
                           raw_str=raw, confidence=0.80)
    except Exception:
        return TypedAnswer(value=raw, answer_type="string",
                           raw_str=raw, confidence=0.1,
                           extra={"parse_error": "fraction"})


def extract_expression(raw: str) -> TypedAnswer:
    sp = _sympy()

    clean = raw
    replacements = [
        (r"\\cdot",                                   "*"),
        (r"\\times",                                  "*"),
        (r"\\div",                                    "/"),
        (r"\\sqrt\s*\{([^}]+)\}",                    r"sqrt(\1)"),
        (r"\\sqrt\s*(\w+)",                           r"sqrt(\1)"),
        (r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}",      r"((\1)/(\2))"),
        (r"\\binom\s*\{([^}]+)\}\s*\{([^}]+)\}",     r"binomial(\1,\2)"),  # IMPROVEMENT 10
        (r"\^",                                       "**"),
        (r"\\left",                                   ""),
        (r"\\right",                                  ""),
        (r"\\pm",                                     ""),
        (r"\{",                                       "("),
        (r"\}",                                       ")"),
        (r"\\pi",                                     "pi"),
        (r"\\infty",                                  "oo"),
        (r"\\log",                                    "log"),
        (r"\\ln",                                     "log"),
        (r"\\sin",                                    "sin"),
        (r"\\cos",                                    "cos"),
        (r"\\tan",                                    "tan"),
    ]
    for pat, repl in replacements:
        clean = re.sub(pat, repl, clean)
    clean = clean.strip()

    try:
        expr = sp.parse_expr(clean, evaluate=True)
        expr = sp.simplify(expr)
        return TypedAnswer(value=expr, answer_type="expression",
                           raw_str=raw, confidence=0.88)
    except Exception:
        return TypedAnswer(value=raw, answer_type="string",
                           raw_str=raw, confidence=0.3,
                           extra={"parse_error": "expression", "clean": clean})


def extract_set(raw: str) -> TypedAnswer:
    sp = _sympy()

    inner = raw.strip()
    inner = re.sub(r"^\\?\{", "", inner)
    inner = re.sub(r"\\?\}$", "", inner)
    inner = inner.replace(" and ", ",").replace(" or ", ",")

    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if not parts:
        return TypedAnswer(value=frozenset(), answer_type="set",
                           raw_str=raw, confidence=0.5)

    parsed = []
    all_ok = True
    for p in parts:
        try:
            parsed.append(sp.parse_expr(p, evaluate=True))
        except Exception:
            parsed.append(p.strip().lower())
            all_ok = False

    confidence = 0.90 if all_ok else 0.65
    return TypedAnswer(value=frozenset(str(x) for x in parsed),
                       answer_type="set",
                       raw_str=raw,
                       confidence=confidence)


def extract_string(raw: str) -> TypedAnswer:
    # IMPROVEMENT 8: normalise LaTeX text macros before lowercasing
    clean = raw.strip()
    clean = re.sub(r"\\text\s*\{([^}]+)\}",   r"\1", clean)
    clean = re.sub(r"\\mathrm\s*\{([^}]+)\}", r"\1", clean)
    clean = re.sub(r"\\textbf\s*\{([^}]+)\}", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean).strip().lower()
    return TypedAnswer(value=clean,
                       answer_type="string",
                       raw_str=raw,
                       confidence=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Master extraction entry point
# ─────────────────────────────────────────────────────────────────────────────

_DETECTOR = AnswerTypeDetector()

_EXTRACTOR_MAP = {
    "integer":    extract_integer,
    "float":      extract_float,
    "fraction":   extract_fraction,
    "expression": extract_expression,
    "set":        extract_set,
    "string":     extract_string,
}


def extract_answer(
    llm_output:  str,
    problem:     str,
    attempt_idx: int = -1,
    forced_type: Optional[str] = None,
) -> Optional[TypedAnswer]:
    """
    Full extraction pipeline:
      1. Pull raw boxed string from LLM output
      2. Detect answer type (or use forced_type)
      3. Run type-specific extractor
      4. Return TypedAnswer (or None if nothing found)
    """
    raw = extract_raw_boxed(llm_output)
    if raw is None:
        return None

    ans_type = forced_type or _DETECTOR.detect(problem, raw)
    extractor = _EXTRACTOR_MAP.get(ans_type, extract_string)
    typed = extractor(raw)
    typed.attempt_idx = attempt_idx
    return typed


# ─────────────────────────────────────────────────────────────────────────────
# Type-aware comparator
# ─────────────────────────────────────────────────────────────────────────────

def answers_match(a: TypedAnswer, b: TypedAnswer) -> bool:
    """
    Return True if two TypedAnswers represent the same mathematical value.

    IMPROVEMENT 6: cross-type integer vs expression comparison.
    """
    numeric = {"integer", "float", "fraction"}

    # ── Both numeric ──────────────────────────────────────────────────────────
    if a.answer_type in numeric and b.answer_type in numeric:
        if a.answer_type == "integer" and b.answer_type == "integer":
            return a.value == b.value
        fa, fb = a.as_float(), b.as_float()
        if fa is None or fb is None:
            return False
        tol = max(a.tolerance, b.tolerance)
        return abs(fa - fb) <= tol

    # ── Both expressions ──────────────────────────────────────────────────────
    if a.answer_type == "expression" and b.answer_type == "expression":
        sp = _sympy()
        try:
            diff = sp.simplify(a.value - b.value)
            return diff == 0
        except Exception:
            try:
                return str(sp.simplify(a.value)) == str(sp.simplify(b.value))
            except Exception:
                return str(a.value) == str(b.value)

    # ── Both sets ─────────────────────────────────────────────────────────────
    if a.answer_type == "set" and b.answer_type == "set":
        if len(a.value) != len(b.value):
            return False
        return a.value == b.value

    # ── Both strings ──────────────────────────────────────────────────────────
    if a.answer_type == "string" and b.answer_type == "string":
        return str(a.value).strip().lower() == str(b.value).strip().lower()

    # ── fraction vs expression ────────────────────────────────────────────────
    if {a.answer_type, b.answer_type} <= {"fraction", "expression"}:
        sp = _sympy()
        try:
            diff = sp.simplify(sp.sympify(a.value) - sp.sympify(b.value))
            return diff == 0
        except Exception:
            return False

    # IMPROVEMENT 6: integer vs expression cross-comparison
    if {a.answer_type, b.answer_type} == {"integer", "expression"}:
        sp = _sympy()
        try:
            int_ans  = a.value if a.answer_type == "integer" else b.value
            expr_ans = b.value if a.answer_type == "integer" else a.value
            simplified = sp.simplify(sp.sympify(expr_ans))
            return int(simplified) == int_ans
        except Exception:
            return False

    # ── numeric vs expression (float/fraction side) ───────────────────────────
    if a.answer_type in numeric and b.answer_type == "expression":
        sp = _sympy()
        try:
            return sp.simplify(sp.sympify(a.value) - sp.sympify(b.value)) == 0
        except Exception:
            return False
    if b.answer_type in numeric and a.answer_type == "expression":
        sp = _sympy()
        try:
            return sp.simplify(sp.sympify(a.value) - sp.sympify(b.value)) == 0
        except Exception:
            return False

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Answer vote selector
# ─────────────────────────────────────────────────────────────────────────────

def select_best_answer(
    candidates: list[TypedAnswer],
    min_votes:  int = 2,
) -> Optional[TypedAnswer]:
    """
    Select the best answer from a list of TypedAnswer objects.

    IMPROVEMENT 9: confidence-weighted group score for tiebreaks.
    Score = sum(confidence) + 0.1 * count
    """
    if not candidates:
        return None

    valid = [c for c in candidates if c is not None and c.confidence >= 0.3]
    if not valid:
        return None

    # Group by mathematical equivalence
    groups: list[list[TypedAnswer]] = []
    for ans in valid:
        placed = False
        for group in groups:
            if answers_match(ans, group[0]):
                group.append(ans)
                placed = True
                break
        if not placed:
            groups.append([ans])

    # IMPROVEMENT 9: score = sum(confidence) + 0.1 * count
    def group_score(g):
        return sum(x.confidence for x in g) + 0.1 * len(g)

    groups.sort(key=group_score, reverse=True)

    best_group = groups[0]
    if len(best_group) >= min_votes:
        return max(best_group, key=lambda x: x.confidence)

    return max(valid, key=lambda x: x.confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("Find the number of integers",        "\\boxed{42}",                              "integer"),
        ("Find the probability",               "\\boxed{0.375}",                           "float"),
        ("Express as a fraction",              "\\boxed{3/4}",                             "fraction"),
        ("Simplify the expression",            "\\boxed{x^2 + 1}",                         "expression"),
        ("Find all integer solutions",         "\\boxed{-1, 2, 5}",                        "set"),
        ("State the theorem",                  "\\boxed{Fermat's last}",                    "string"),
        # Fix 1 — trailing period
        ("Find the sum",                       "\\boxed{-54.}",                            "integer"),
        # Fix 2 — expression label but plain integer in box
        ("zeros of the polynomial P(x)=x^4",  "\\boxed{16}",                              "integer"),
        # Fix 3 — lone period (should return None)
        ("What is the parity",                 "assistant\nThe answer is.",                None),
        # Fix 4 — degree symbol
        ("Find all possible angle values",     "\\boxed{0^\\circ \\text{ and } 360^\\circ}", "string"),
        # Fix 5 — nested braces
        ("Express as simplified fraction",     "\\boxed{\\frac{3}{4}}",                   "fraction"),
        # Fix 7 — parity/string answer
        ("Determine whether n is even or odd", "\\boxed{n \\text{ is even}}",              "string"),
        # Fix 8 — LaTeX text macro
        ("Find all n satisfying",              "\\boxed{n \\text{ is even}}",              "string"),
    ]

    print("\n=== Answer type extraction smoke test ===\n")
    passed = failed = 0
    for prob, boxed_or_output, expected in cases:
        fake = boxed_or_output if boxed_or_output.startswith("assistant") \
               else f"assistant\n{boxed_or_output}"
        ta = extract_answer(fake, prob)
        if expected is None:
            ok = (ta is None)
        else:
            ok = (ta is not None and ta.answer_type == expected)
        status = "OK  " if ok else "FAIL"
        if ok: passed += 1
        else:  failed += 1
        got = "None" if ta is None else repr(ta)
        print(f"  [{status}]  expected={str(expected):12s}  got={got}")
    print(f"\n  {passed}/{passed+failed} passed\n")
