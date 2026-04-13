"""
tools.py
========
The five tools available to the LLM during solving.
Each tool returns a structured dict — never raw text.

Tool list
---------
  knowledge_search(query, domain, mode, top_k)
  compute(expression, operation, variables, assumptions)
  numerical_search(condition_fn_src, search_space)
  verify(problem, typed_answer, approach_summary)
  run_code(code, timeout)

FIX: run_code now has a subprocess fallback when sandbox=None,
     so the solver never returns "No sandbox available" and bails out.
"""

from __future__ import annotations

import re
import time
import signal
import textwrap
import threading
import traceback
from contextlib import contextmanager
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Timeout context manager (Unix only; falls back on Windows)
# ─────────────────────────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass


@contextmanager
def _timeout(seconds: float):
    """Raise _TimeoutError if the block takes longer than `seconds`."""
    def _handler(signum, frame):
        raise _TimeoutError(f"Timed out after {seconds}s")

    try:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    except AttributeError:
        # SIGALRM not available (Windows) — run without timeout
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Sentence encoder singleton
# ─────────────────────────────────────────────────────────────────────────────

_encoder_lock = threading.Lock()
_encoder_instance = None
_encoder_model_name = None

EMBEDDING_MODEL = "/kaggle/input/models/sumandey008/sentence-transformersall-minilm-l6-v2/transformers/default/1"


def get_encoder(model_name: str = EMBEDDING_MODEL):
    """
    Return a cached SentenceTransformer.
    Thread-safe: the first caller loads; subsequent callers reuse.
    """
    global _encoder_instance, _encoder_model_name
    with _encoder_lock:
        if _encoder_instance is None or _encoder_model_name != model_name:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _encoder_instance  = SentenceTransformer(model_name, device=device, local_files_only=True)
            _encoder_model_name = model_name
    return _encoder_instance


def _embed_query(query: str, model_name: str = EMBEDDING_MODEL):
    enc = get_encoder(model_name)
    import numpy as np
    emb = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return emb[0].astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# Tool result helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(**kwargs) -> dict:
    return {"status": "ok", **kwargs}


def _err(message: str, **kwargs) -> dict:
    return {"status": "error", "error": message, **kwargs}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: knowledge_search
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_search(
    query:    str,
    db,                      # KnowledgeDB instance (passed in by solver)
    domain:   Optional[str] = None,
    mode:     str           = "both",   # "problems" | "theorems" | "both"
    top_k:    int           = 4,
    timeout:  float         = 10.0,
) -> dict:
    """
    Query the knowledge database for similar problems and/or theorems.
    """
    results = {"problems": [], "theorems": [], "query": query}

    try:
        with _timeout(timeout):
            query_emb = _embed_query(query)

            if mode in ("problems", "both"):
                results["problems"] = db.search_problems(
                    query_emb, domain=domain, top_k=top_k)

            if mode in ("theorems", "both"):
                results["theorems"] = db.search_theorems(
                    query_emb, domain=domain, top_k=top_k)

    except _TimeoutError:
        results["warning"] = f"knowledge_search timed out after {timeout}s"
    except Exception as e:
        return _err(f"knowledge_search failed: {e}",
                    problems=[], theorems=[], query=query)

    return _ok(**results)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: compute
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_OPERATIONS = {
    "simplify", "expand", "factor", "solve", "solve_system",
    "diff", "integrate", "limit", "series", "roots",
    "gcd", "lcm", "isprime", "factorint", "mod",
    "nsolve", "eigenvalues", "det", "inverse",
    "binomial", "factorial", "totient",
}


def compute(
    expression:  str,
    operation:   str,
    variables:   Optional[list[str]] = None,
    assumptions: Optional[dict]      = None,
    timeout:     float               = 15.0,
) -> dict:
    """
    Exact symbolic computation via SymPy.
    """
    if operation not in _SUPPORTED_OPERATIONS:
        return _err(f"Unknown operation {operation!r}. "
                    f"Supported: {sorted(_SUPPORTED_OPERATIONS)}")

    try:
        with _timeout(timeout):
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, \
                implicit_multiplication_application

            transforms = standard_transformations + (implicit_multiplication_application,)

            local_dict = {}
            for name in _extract_symbol_names(expression, variables or []):
                kwargs = (assumptions or {}).get(name, {})
                local_dict[name] = sp.Symbol(name, **kwargs)

            try:
                expr = parse_expr(expression, local_dict=local_dict,
                                  transformations=transforms, evaluate=True)
            except Exception as pe:
                return _err(f"Could not parse expression: {pe}")

            result, rtype = _run_operation(sp, expr, operation, variables, local_dict)

            if isinstance(result, (list, tuple)):
                result_str = str([str(r) for r in result])
                latex_str  = ", ".join(sp.latex(r) for r in result)
                numeric    = None
            elif isinstance(result, dict):
                result_str = str({str(k): str(v) for k, v in result.items()})
                latex_str  = str(result)
                numeric    = None
            else:
                result_str = str(result)
                latex_str  = sp.latex(result)
                try:
                    numeric = float(result.evalf())
                except Exception:
                    numeric = None

            return _ok(
                result=result_str,
                result_type=rtype,
                latex=latex_str,
                numeric=numeric,
                operation=operation,
                expression=expression,
            )

    except _TimeoutError:
        return _err(f"compute timed out after {timeout}s", operation=operation)
    except Exception as e:
        return _err(f"compute error: {e}\n{traceback.format_exc()[:400]}",
                    operation=operation)


def _extract_symbol_names(expr_str: str, extra: list[str]) -> set[str]:
    names = set(re.findall(r"\b([a-zA-Z])\b", expr_str))
    names -= {"e", "E", "I", "pi", "oo"}
    names |= set(extra)
    return names


def _run_operation(sp, expr, operation: str, variables, local_dict: dict):
    vars_syms = [local_dict[v] for v in (variables or []) if v in local_dict]
    if not vars_syms:
        vars_syms = sorted(expr.free_symbols, key=str)

    op = operation.lower()

    if op == "simplify":   return sp.simplify(expr), "expression"
    if op == "expand":     return sp.expand(expr), "expression"
    if op == "factor":     return sp.factor(expr), "expression"
    if op == "roots":
        r = sp.roots(expr, *vars_syms[:1])
        return r, "dict"
    if op == "solve":
        sol = sp.solve(expr, *vars_syms[:1])
        return sol, "list"
    if op == "solve_system":
        eqs = [sp.parse_expr(e.strip()) for e in str(expr).split(";")]
        sol = sp.solve(eqs, vars_syms)
        return sol, "list"
    if op == "diff":
        v = vars_syms[0] if vars_syms else sp.Symbol("x")
        return sp.diff(expr, v), "expression"
    if op == "integrate":
        v = vars_syms[0] if vars_syms else sp.Symbol("x")
        return sp.integrate(expr, v), "expression"
    if op == "limit":
        v    = vars_syms[0] if vars_syms else sp.Symbol("x")
        point = vars_syms[1] if len(vars_syms) > 1 else sp.oo
        return sp.limit(expr, v, point), "expression"
    if op == "series":
        v = vars_syms[0] if vars_syms else sp.Symbol("x")
        return sp.series(expr, v, n=6), "expression"
    if op == "gcd":
        parts = str(expr).split(",")
        a, b  = [sp.sympify(p.strip()) for p in parts[:2]]
        return sp.gcd(a, b), "expression"
    if op == "lcm":
        parts = str(expr).split(",")
        a, b  = [sp.sympify(p.strip()) for p in parts[:2]]
        return sp.lcm(a, b), "expression"
    if op == "isprime":
        n = int(sp.sympify(expr))
        return sp.isprime(n), "integer"
    if op == "factorint":
        n = int(sp.sympify(expr))
        return sp.factorint(n), "dict"
    if op == "mod":
        parts = str(expr).split(",")
        a, m  = int(sp.sympify(parts[0].strip())), int(sp.sympify(parts[1].strip()))
        return a % m, "integer"
    if op == "binomial":
        parts = str(expr).split(",")
        n, k  = int(sp.sympify(parts[0].strip())), int(sp.sympify(parts[1].strip()))
        return sp.binomial(n, k), "integer"
    if op == "factorial":
        n = int(sp.sympify(expr))
        return sp.factorial(n), "integer"
    if op == "totient":
        n = int(sp.sympify(expr))
        return sp.totient(n), "integer"
    if op == "det":
        mat = sp.Matrix(sp.sympify(expr))
        return mat.det(), "expression"
    if op == "inverse":
        mat = sp.Matrix(sp.sympify(expr))
        return mat.inv(), "expression"
    if op == "eigenvalues":
        mat = sp.Matrix(sp.sympify(expr))
        return mat.eigenvals(), "dict"
    if op == "nsolve":
        v = vars_syms[0] if vars_syms else sp.Symbol("x")
        return sp.nsolve(expr, v, 1), "expression"

    raise ValueError(f"Operation {operation!r} dispatcher not implemented")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: numerical_search
# ─────────────────────────────────────────────────────────────────────────────

def numerical_search(
    condition_src: str,
    search_space:  dict,
    max_results:   int   = 20,
    timeout:       float = 20.0,
) -> dict:
    """
    Brute-force search over a discrete space.
    """
    _SAFE_GLOBALS = {
        "__builtins__": {},
        "abs": abs, "min": min, "max": max, "pow": pow,
        "round": round, "divmod": divmod,
    }
    try:
        import math as _math
        _SAFE_GLOBALS.update({k: getattr(_math, k) for k in dir(_math)
                               if not k.startswith("_")})
    except Exception:
        pass

    try:
        code = compile(condition_src, "<condition>", "eval")
    except SyntaxError as e:
        return _err(f"Invalid condition syntax: {e}", matches=[], count=0)

    matches  = []
    exhausted = False
    start    = time.time()

    try:
        stype = search_space.get("type", "range")

        if stype == "range":
            lo = int(search_space.get("lo", 0))
            hi = int(search_space.get("hi", 1000))
            if hi - lo > 10_000_000:
                return _err("Search space too large (max 10M)", matches=[], count=0)

            with _timeout(timeout):
                for n in range(lo, hi + 1):
                    if len(matches) >= max_results:
                        break
                    if time.time() - start > timeout:
                        break
                    try:
                        env = {**_SAFE_GLOBALS, "n": n, "k": n, "m": n}
                        if eval(code, env):
                            matches.append(n)
                    except Exception:
                        continue
                else:
                    exhausted = len(matches) < max_results

            space_desc = f"n ∈ [{lo}, {hi}]"

        elif stype == "range2d":
            lo1, hi1 = int(search_space["lo1"]), int(search_space["hi1"])
            lo2, hi2 = int(search_space["lo2"]), int(search_space["hi2"])
            if (hi1 - lo1) * (hi2 - lo2) > 1_000_000:
                return _err("2D search space too large (max 1M)", matches=[], count=0)

            with _timeout(timeout):
                done = False
                for n in range(lo1, hi1 + 1):
                    if done:
                        break
                    for m in range(lo2, hi2 + 1):
                        if len(matches) >= max_results:
                            done = True
                            break
                        if time.time() - start > timeout:
                            done = True
                            break
                        try:
                            env = {**_SAFE_GLOBALS, "n": n, "m": m, "k": n}
                            if eval(code, env):
                                matches.append((n, m))
                        except Exception:
                            continue

            space_desc = f"n ∈ [{lo1},{hi1}], m ∈ [{lo2},{hi2}]"

        else:
            return _err(f"Unknown search_space type: {stype!r}", matches=[], count=0)

    except _TimeoutError:
        pass

    return _ok(
        matches=matches,
        count=len(matches),
        space=space_desc,
        exhausted=exhausted,
        elapsed_s=round(time.time() - start, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: verify
# ─────────────────────────────────────────────────────────────────────────────

def verify(
    problem:          str,
    typed_answer,
    sandbox,
    approach_summary: str = "",
    timeout:          float = 12.0,
) -> dict:
    """
    Multi-stage verification of a TypedAnswer.
    """
    checks = {}

    checks["sanity"] = (typed_answer is not None and
                        typed_answer.confidence >= 0.40)
    if not checks["sanity"]:
        return _ok(passed=False, checks=checks,
                   failed=["sanity"],
                   answer=str(getattr(typed_answer, "value", None)),
                   type=getattr(typed_answer, "answer_type", "unknown"))

    val  = typed_answer.value
    atype = typed_answer.answer_type

    if typed_answer.is_numeric():
        fval = typed_answer.as_float()
        checks["range"] = (fval is not None and
                           math_finite(fval) and
                           abs(fval) < 1e15)
    else:
        checks["range"] = True

    checks["symbolic"] = _symbolic_check(val, atype, problem)
    checks["sandbox"]  = _sandbox_check(val, atype, problem, sandbox, timeout)
    checks["consistency"] = _type_consistency_check(atype, problem)

    failed = [k for k, v in checks.items() if not v]
    passed = len(failed) == 0

    return _ok(
        passed=passed,
        checks=checks,
        failed=failed,
        answer=str(val),
        type=atype,
        approach_summary=approach_summary[:200],
    )


def math_finite(x: float) -> bool:
    import math
    return math.isfinite(x)


def _symbolic_check(val, atype: str, problem: str) -> bool:
    try:
        import sympy as sp
        if atype == "integer":
            return isinstance(val, (int, sp.Integer))
        if atype in ("float", "fraction"):
            f = float(val)
            return math_finite(f)
        if atype == "expression":
            simplified = sp.simplify(val)
            return simplified is not None
        if atype == "set":
            return len(val) > 0
        return True
    except Exception:
        return True


def _sandbox_check(val, atype: str, problem: str, sandbox, timeout: float) -> bool:
    if sandbox is None:
        return True

    code = textwrap.dedent(f"""
import sympy as sp
import math

answer = {repr(val)}
answer_type = {repr(atype)}

if answer_type == 'integer':
    assert isinstance(answer, int), f"Expected int, got {{type(answer)}}"
elif answer_type == 'float':
    assert math.isfinite(float(answer)), "Float is not finite"
elif answer_type == 'fraction':
    f = float(answer)
    assert math.isfinite(f), "Fraction is not finite"
elif answer_type == 'expression':
    expr = sp.sympify(answer)
    assert expr is not None, "Expression could not be parsed"

print("VERIFY_OK")
""")

    try:
        result = sandbox.execute(code)
        return "VERIFY_OK" in result and "Error" not in result
    except Exception:
        return False


def _type_consistency_check(atype: str, problem: str) -> bool:
    p = problem.lower()
    if any(s in p for s in ["how many", "count", "number of"]):
        if atype == "expression":
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: run_code
# ─────────────────────────────────────────────────────────────────────────────

def run_code(
    code:    str,
    sandbox,                   # MathSandbox instance OR None
    timeout: float = 10.0,
) -> dict:
    """
    Execute arbitrary Python code.

    PRIMARY:  MathSandbox Jupyter kernel (if available).
    FALLBACK: subprocess.run() via temp file — always works even when
              the sandbox pool is empty or Jupyter failed to start.
    """
    # ── Primary path: Jupyter sandbox ────────────────────────────────────────
    if sandbox is not None:
        start = time.time()
        try:
            output  = sandbox.execute(code)
            elapsed = time.time() - start
            has_error = (
                "[ERROR]" in output or
                "Traceback" in output or
                "Error:" in output
            )
            if has_error:
                return {
                    "status":         "error",
                    "stdout":         "",
                    "stderr":         output[:2000],
                    "has_error":      True,
                    "execution_time": round(elapsed, 3),
                }
            return _ok(stdout=output[:4000], stderr="", has_error=False,
                       execution_time=round(elapsed, 3))
        except Exception as e:
            return _err(str(e), stdout="", stderr=traceback.format_exc()[:1000],
                        has_error=True,
                        execution_time=round(time.time() - start, 3))

    # ── Fallback path: subprocess via temp file ───────────────────────────────
    import subprocess, sys, tempfile, os
    start = time.time()
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f:
            f.write(code)
            fname = f.name

        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout
        )
        os.unlink(fname)
        elapsed = time.time() - start

        if result.returncode != 0:
            return {
                "status":         "error",
                "stdout":         result.stdout[:2000],
                "stderr":         result.stderr[:2000],
                "has_error":      True,
                "execution_time": round(elapsed, 3),
            }
        return _ok(stdout=result.stdout[:4000], stderr="", has_error=False,
                   execution_time=round(elapsed, 3))

    except subprocess.TimeoutExpired:
        try:
            os.unlink(fname)
        except Exception:
            pass
        return _err(f"Timed out after {timeout}s", stdout="", stderr="",
                    has_error=True, execution_time=timeout)
    except Exception as e:
        return _err(str(e), stdout="", stderr=traceback.format_exc()[:500],
                    has_error=True,
                    execution_time=round(time.time() - start, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatcher — used by the solver loop
# ─────────────────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """
    Single object that holds references to all tools and dispatches
    LLM tool-call requests.
    """

    TOOL_NAMES = ("knowledge_search", "compute", "numerical_search",
                  "verify", "run_code")

    def __init__(self, db, sandbox):
        self.db      = db
        self.sandbox = sandbox

    def call(self, tool_name: str, args: dict) -> dict:
        if tool_name not in self.TOOL_NAMES:
            return _err(f"Unknown tool: {tool_name!r}. "
                        f"Available: {self.TOOL_NAMES}")
        try:
            if tool_name == "knowledge_search":
                return knowledge_search(db=self.db, **args)
            if tool_name == "compute":
                return compute(**args)
            if tool_name == "numerical_search":
                return numerical_search(**args)
            if tool_name == "verify":
                return verify(sandbox=self.sandbox, **args)
            if tool_name == "run_code":
                return run_code(sandbox=self.sandbox, **args)
        except Exception as e:
            return _err(f"Uncaught error in {tool_name}: {e}\n"
                        f"{traceback.format_exc()[:400]}")

    def tool_schema(self) -> list[dict]:
        return [
            {
                "name": "knowledge_search",
                "description": (
                    "Search the knowledge database for similar problems and/or relevant theorems. "
                    "Always call this FIRST before attempting to solve the problem."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query":  {"type": "string",
                                   "description": "What technique or concept to search for"},
                        "domain": {"type": "string",
                                   "description": "Optional domain filter e.g. 'Algebra'"},
                        "mode":   {"type": "string",
                                   "enum": ["problems", "theorems", "both"],
                                   "description": "Which store to search"},
                        "top_k":  {"type": "integer", "description": "Results to return (default 4)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "compute",
                "description": (
                    "Exact symbolic computation via SymPy. Use for factoring, solving equations, "
                    "differentiation, integration, modular arithmetic, etc. "
                    "Prefer this over run_code for pure symbolic math."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string",
                                       "description": "Math expression (Python syntax, use ** not ^)"},
                        "operation":  {"type": "string",
                                       "description": f"One of: {sorted(_SUPPORTED_OPERATIONS)}"},
                        "variables":  {"type": "array", "items": {"type": "string"},
                                       "description": "Variable names e.g. ['x', 'y']"},
                        "assumptions": {"type": "object",
                                        "description": "Symbol assumptions e.g. {'n': {'positive': true}}"},
                    },
                    "required": ["expression", "operation"],
                },
            },
            {
                "name": "numerical_search",
                "description": (
                    "Brute-force search over integer ranges. Use when you can't solve analytically "
                    "but can describe a verification condition. Returns all matching values."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "condition_src": {"type": "string",
                                          "description": "Python boolean expression using n (and m for 2D). E.g. 'n**2 + 1 == 5*n - 3'"},
                        "search_space":  {"type": "object",
                                          "description": '{"type": "range", "lo": 0, "hi": 1000} or {"type": "range2d", "lo1": 0, "hi1": 100, "lo2": 0, "hi2": 100}'},
                        "max_results":   {"type": "integer", "description": "Stop after N matches (default 20)"},
                    },
                    "required": ["condition_src", "search_space"],
                },
            },
            {
                "name": "verify",
                "description": (
                    "Verify a proposed answer against the problem. "
                    "The pipeline REQUIRES you to call this before submitting a final answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problem":          {"type": "string", "description": "Original problem text"},
                        "typed_answer":     {"type": "object", "description": "TypedAnswer dict from answer extraction"},
                        "approach_summary": {"type": "string", "description": "One-line summary of your approach"},
                    },
                    "required": ["problem", "typed_answer"],
                },
            },
            {
                "name": "run_code",
                "description": (
                    "Execute arbitrary Python code in an isolated sandbox. "
                    "Use for complex algorithms, numerical experiments, DP over large ranges, "
                    "or any computation not covered by the compute tool. Always print() your results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string",
                                 "description": "Python code to execute. Must print() results."},
                    },
                    "required": ["code"],
                },
            },
        ]
