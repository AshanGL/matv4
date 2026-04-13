"""
solver.py
=========
OlympiadSolver — semi-agentic tool-based solver.

Architecture
------------
  Pipeline controller  (this file)
      │
      ├─ Phase 1: forced knowledge_search
      ├─ Phase 2: LLM ↔ tools loop (up to max_turns)
      ├─ Phase 3: forced verify
      └─ Phase 4: vote and select answer

KEY FIXES vs prior version
--------------------------
1. AnswerTypeInference loads encoder from config.json["encoder_name"]
   (the local DeBERTa path), then loads head weights on top.
   Never calls AutoModel.from_pretrained(ckpt_dir) which fails when
   ckpt_dir only has head weights.
2. Tokenizer loaded from encoder_name path, not ckpt_dir — fixes the
   Mistral tokenizer warning.
3. SYSTEM_PROMPT massively improved: explicit tool-call JSON format,
   type-specific answer formatting rules, expression/fraction examples.
4. Fallback answer extraction: if LLM produces no \\boxed{}, we scan
   for the last integer / fraction / expression in the response.
5. Smarter "no answer" recovery: re-prompt with a direct extraction
   message rather than immediately returning 0.
6. per-attempt temperature variation for better diversity.
"""

from __future__ import annotations

import gc
import os
import re
import json
import time
import math
import queue
import threading
import traceback
import contextlib
import warnings
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
warnings.filterwarnings("ignore")

from answer_types import (
    TypedAnswer, extract_answer, select_best_answer, answers_match, ANSWER_TYPES
)
from tools import ToolDispatcher, run_code, verify
from knowledge_db import KnowledgeDB
from prompts import (
    build_system_prompt, build_user_prompt, build_extraction_nudge,
    build_retry_prompt, TOOL_CALL_FORMAT_REMINDER
)

# ─────────────────────────────────────────────────────────────────────────────
# Solver configuration
# ─────────────────────────────────────────────────────────────────────────────

class SolverConfig:
    # Model
    model_path           = "/kaggle/input/gpt-oss-120b/transformers/default/1"
    served_model_name    = "gpt-oss"

    # Memory
    kv_cache_dtype       = "fp8_e4m3"
    dtype                = "auto"
    gpu_memory_utilization = 0.92

    # Generation
    max_tokens           = 16384
    temperature          = 0.7        # base; each attempt varies ±0.15
    min_p                = 0.02
    top_logprobs         = 5
    context_tokens       = 131072
    stream_interval      = 128

    # Timing
    notebook_limit       = 17400
    server_timeout       = 180
    high_problem_timeout = 900
    base_problem_timeout = 300
    sandbox_timeout      = 5
    jupyter_timeout      = 10

    # Concurrency
    attempts             = 6
    workers              = 6
    max_tool_turns       = 14
    early_stop_votes     = 4
    seed                 = 42
    batch_size           = 64

    # Paths
    ckpt_dir             = "/kaggle/working/checkpoints"
    db_dir               = "/kaggle/working/knowledge_db"


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight inference wrappers for trained classifiers
# ─────────────────────────────────────────────────────────────────────────────

class AnswerTypeInference:
    """
    Load trained AnswerTypeClassifier for CPU inference.

    LOADING STRATEGY (fixed):
    - config.json["encoder_name"] points to the local DeBERTa folder
      (set by training and optionally patched by patch_local_paths.py)
    - We load the encoder from that path (which HAS model weights)
    - We load the head from head_weights.pt (small, fast)
    - Tokenizer also loaded from encoder_name path to avoid stale
      tokenizer files from other models in ckpt_dir
    """

    def __init__(self, ckpt_dir: str):
        from transformers import AutoTokenizer
        from train_new import AnswerTypeModel, ANSWER_TYPES

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg = json.load(f)

        encoder_name = cfg.get("encoder_name", "microsoft/deberta-v3-base")

        # Tokenizer from the encoder path (never from ckpt_dir to avoid stale files)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        # Build model architecture, load encoder weights from encoder_name
        self.model = AnswerTypeModel(encoder_name, n_classes=len(ANSWER_TYPES))

        # Load fine-tuned weights.  Prefer full best_model.pt; fall back to head only.
        full_path = os.path.join(ckpt_dir, "best_model.pt")
        head_path = os.path.join(ckpt_dir, "head_weights.pt")

        if os.path.exists(full_path):
            state = torch.load(full_path, map_location="cpu")
            # best_model.pt may be full model state dict or just head weights
            # Detect by checking if encoder keys are present
            if any(k.startswith("encoder.") for k in state):
                self.model.load_state_dict(state)
            else:
                # Only head weights — that's fine, encoder already loaded above
                self.model.head.load_state_dict(state)
        elif os.path.exists(head_path):
            self.model.head.load_state_dict(
                torch.load(head_path, map_location="cpu"))
        else:
            raise FileNotFoundError(
                f"No weights found in {ckpt_dir}. "
                f"Expected best_model.pt or head_weights.pt")

        self.model.eval()
        self.answer_types = cfg.get("answer_types", ANSWER_TYPES)

    def predict(self, problem: str) -> str:
        enc = self.tokenizer(problem, max_length=256, padding="max_length",
                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
        return self.answer_types[logits.argmax(-1).item()]


class VerifyScorerInference:
    """Load trained VerifyScorer for CPU inference."""

    def __init__(self, ckpt_dir: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg = json.load(f)

        encoder_name = cfg.get("encoder_name", "microsoft/deberta-v3-base")

        # Load tokenizer from encoder path (not ckpt_dir) to avoid stale files
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        # Model weights are fully inside ckpt_dir (save_pretrained was called)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir, num_labels=2)
        self.model.eval()

    def score(self, problem: str, trace: str) -> tuple[bool, float]:
        enc = self.tokenizer(problem, trace, max_length=384, padding="max_length",
                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"]).logits
        probs    = torch.softmax(logits, -1).squeeze()
        is_valid = logits.argmax(-1).item() == 1
        return is_valid, float(probs[1])


# ─────────────────────────────────────────────────────────────────────────────
# Tool call parser
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)
_JSON_BLOCK_PATTERN = re.compile(
    r'```(?:json)?\s*(\{[^`]*?"name"\s*:\s*"(?:knowledge_search|compute|'
    r'numerical_search|verify|run_code)"[^`]*?\})\s*```',
    re.DOTALL,
)
# Bare JSON objects that look like tool calls (without wrappers)
_BARE_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(knowledge_search|compute|numerical_search|verify|run_code)"'
    r'.*?"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    seen  = set()

    def _add(obj):
        key = json.dumps(obj, sort_keys=True)
        if key not in seen:
            seen.add(key)
            calls.append({"name": obj["name"],
                          "arguments": obj.get("arguments", obj.get("args", {}))})

    for pat in (_TOOL_CALL_PATTERN, _JSON_BLOCK_PATTERN):
        for match in pat.finditer(text):
            try:
                obj = json.loads(match.group(1))
                if "name" in obj and obj["name"] in ToolDispatcher.TOOL_NAMES:
                    _add(obj)
            except json.JSONDecodeError:
                pass

    for match in _BARE_JSON_PATTERN.finditer(text):
        try:
            name = match.group(1)
            args_str = match.group(2)
            args = json.loads(args_str)
            _add({"name": name, "arguments": args})
        except Exception:
            pass

    return calls


def format_tool_result(tool_name: str, result: dict) -> str:
    status = result.get("status", "?")
    if tool_name == "knowledge_search":
        theorems = result.get("theorems", [])
        problems = result.get("problems", [])
        lines = [f"[knowledge_search — {status}]"]
        if theorems:
            lines.append("Theorems found:")
            for t in theorems[:4]:
                lines.append(f"  • {t['name']} (sim={t.get('similarity',0):.2f})")
                lines.append(f"    When to use: {t['when_to_apply']}")
                if t.get("statement"):
                    lines.append(f"    Statement: {t['statement'][:200]}")
        if problems:
            lines.append("Similar problems (technique hints only):")
            for p in problems[:3]:
                tags = ", ".join(p.get("technique_tags", [])[:5])
                lines.append(f"  • [{p.get('domain','')} / {p.get('difficulty_band','')}]"
                             f"  tags: {tags}")
        return "\n".join(lines)

    if tool_name == "compute":
        if status == "ok":
            return (f"[compute — ok]\n"
                    f"  operation : {result.get('operation')}\n"
                    f"  result    : {result.get('result')}\n"
                    f"  latex     : {result.get('latex')}\n"
                    f"  numeric   : {result.get('numeric')}")
        return f"[compute — error] {result.get('error')}"

    if tool_name == "numerical_search":
        if status == "ok":
            return (f"[numerical_search — ok]\n"
                    f"  space   : {result.get('space')}\n"
                    f"  matches : {result.get('matches')}\n"
                    f"  count   : {result.get('count')}")
        return f"[numerical_search — error] {result.get('error')}"

    if tool_name == "verify":
        if status == "ok":
            return (f"[verify — {'PASSED' if result.get('passed') else 'FAILED'}]\n"
                    f"  checks : {result.get('checks')}\n"
                    f"  failed : {result.get('failed')}\n"
                    f"  answer : {result.get('answer')}  ({result.get('type')})")
        return f"[verify — error] {result.get('error')}"

    if tool_name == "run_code":
        if status == "ok":
            out = result.get("stdout", "")[:2000]
            return f"[run_code — ok]\n{out}"
        return f"[run_code — error]\n{result.get('stderr','')[:500]}"

    return f"[{tool_name}] {json.dumps(result)[:500]}"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback answer extraction (when LLM gives no \boxed{})
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_extract(text: str, forced_type: Optional[str], problem: str,
                      attempt_idx: int) -> Optional[TypedAnswer]:
    """
    Last-resort extraction when no \\boxed{} found.
    Scans the last 800 chars of assistant output for numeric / expression answers.
    """
    # Restrict to the very end of the response
    tail = text[-800:]

    # 1. "Therefore, the answer is X" / "= X" / "answer: X"
    patterns = [
        r'(?:answer\s+is|therefore|thus|hence|final answer|equals?)[:\s=]*(-?[\d,/\.]+)',
        r'=\s*(-?[\d]+(?:\.\d+)?(?:/[\d]+)?)\s*(?:\.|$)',
        r'\\boxed\s*\{\s*([^}]+)\s*\}',   # may have been missed by primary extractor
    ]
    for pat in patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            return extract_answer(
                f"assistant\n\\boxed{{{raw}}}", problem,
                attempt_idx=attempt_idx, forced_type=forced_type)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(client, system_prompt: str, messages: list[dict],
             cfg: SolverConfig, seed: int = 42, temperature: float = None) -> str:
    temp = temperature if temperature is not None else cfg.temperature
    try:
        from openai_harmony import (
            load_harmony_encoding, HarmonyEncodingName,
            SystemContent, ReasoningEffort,
            Message, Role, Conversation,
        )

        sys_content = (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(ReasoningEffort.HIGH)
        )
        sys_msg   = Message.from_role_and_content(Role.SYSTEM, sys_content)
        conv_msgs = [sys_msg] + [
            Message.from_role_and_content(
                Role.USER if m["role"] == "user" else Role.ASSISTANT,
                m["content"]
            ) for m in messages
        ]

        conversation     = Conversation(messages=conv_msgs)
        encoding         = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids   = encoding.stop_tokens_for_assistant_actions()
        prompt_token_ids = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT)

        chunks = []
        stream = client.completions.create(
            model       = cfg.served_model_name,
            prompt      = prompt_token_ids,
            temperature = temp,
            max_tokens  = cfg.max_tokens,
            seed        = seed,
            stream      = True,
            extra_body  = {
                "min_p":          cfg.min_p,
                "stop_token_ids": stop_token_ids,
            },
        )
        for chunk in stream:
            delta = chunk.choices[0].text
            if delta:
                chunks.append(delta)
        stream.close()
        return "".join(chunks)

    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Single attempt executor
# ─────────────────────────────────────────────────────────────────────────────

def run_attempt(
    problem:       str,
    client,
    dispatcher:    ToolDispatcher,
    cfg:           SolverConfig,
    attempt_idx:   int,
    stop_event:    threading.Event,
    deadline:      float,
    forced_type:   Optional[str] = None,
) -> Optional[TypedAnswer]:
    """
    Run one attempt of the solver loop.
    Returns a TypedAnswer or None.
    """
    if stop_event.is_set() or time.time() > deadline:
        return None

    # Per-attempt temperature variation for diversity
    base_temp  = cfg.temperature
    temp_delta = (attempt_idx - cfg.attempts // 2) * 0.05
    temperature = max(0.3, min(1.2, base_temp + temp_delta))

    seed = (cfg.seed + attempt_idx * 7) % (2**31)

    # Build the system prompt (with type hint if known)
    system_prompt = build_system_prompt(forced_type=forced_type)

    messages: list[dict] = []
    verified_answer: Optional[TypedAnswer] = None
    typed: Optional[TypedAnswer] = None

    try:
        # ── Phase 1: forced knowledge_search ─────────────────────────────────
        ks_result = dispatcher.call("knowledge_search", {
            "query":  problem,
            "mode":   "both",
            "top_k":  5,
        })

        user_msg = build_user_prompt(problem, ks_result, forced_type=forced_type)
        messages.append({"role": "user", "content": user_msg})

        # ── Phase 2: solve loop ───────────────────────────────────────────────
        turn         = 0
        verify_called = False
        empty_turns  = 0  # consecutive turns with no tool call and no answer

        while turn < cfg.max_tool_turns:
            if stop_event.is_set() or time.time() > deadline:
                break

            llm_text = call_llm(client, system_prompt, messages, cfg,
                                 seed=seed + turn, temperature=temperature)
            messages.append({"role": "assistant", "content": llm_text})

            # Try primary boxed extraction
            typed = extract_answer(llm_text, problem, attempt_idx=attempt_idx,
                                   forced_type=forced_type)

            # Try fallback extraction if primary failed
            if typed is None:
                typed = _fallback_extract(llm_text, forced_type, problem, attempt_idx)

            tool_calls = parse_tool_calls(llm_text)

            if not tool_calls:
                if typed is not None:
                    # LLM gave an answer with no tools — accept it
                    break
                empty_turns += 1
                if empty_turns >= 2:
                    # After 2 empty turns with no answer, give a strong nudge
                    nudge = build_extraction_nudge(forced_type)
                    messages.append({"role": "user", "content": nudge})
                    # Try one more LLM call explicitly asking for the answer
                    llm_text2 = call_llm(client, system_prompt, messages, cfg,
                                          seed=seed + turn + 100,
                                          temperature=0.3)  # low temp for extraction
                    messages.append({"role": "assistant", "content": llm_text2})
                    typed = extract_answer(llm_text2, problem,
                                           attempt_idx=attempt_idx,
                                           forced_type=forced_type)
                    if typed is None:
                        typed = _fallback_extract(llm_text2, forced_type, problem, attempt_idx)
                    break
                else:
                    messages.append({
                        "role": "user",
                        "content": build_retry_prompt(turn, forced_type),
                    })
                turn += 1
                continue

            empty_turns = 0  # reset on tool call

            # Execute tool calls
            tool_result_texts = []
            for tc in tool_calls[:3]:
                if stop_event.is_set():
                    break
                tname  = tc["name"]
                targs  = tc.get("arguments", {})
                result = dispatcher.call(tname, targs)
                tool_result_texts.append(format_tool_result(tname, result))

                if tname == "verify" and result.get("passed"):
                    verify_called = True
                    if typed is not None:
                        verified_answer = typed

            messages.append({"role": "user",
                              "content": "\n\n".join(tool_result_texts)})
            turn += 1

        # ── Phase 3: force verify if not called ──────────────────────────────
        if typed is not None and not verify_called:
            verify_result = dispatcher.call("verify", {
                "problem":      problem,
                "typed_answer": {"value":       str(typed.value),
                                 "answer_type": typed.answer_type,
                                 "raw_str":     typed.raw_str,
                                 "confidence":  typed.confidence},
                "approach_summary": "Pipeline-forced final verify",
            })
            if verify_result.get("passed"):
                verified_answer = typed

        return verified_answer if verified_answer is not None else typed

    except Exception as e:
        print(f"  [attempt {attempt_idx}] error: {e}")
        traceback.print_exc()
        return None
    finally:
        del messages


# ─────────────────────────────────────────────────────────────────────────────
# OlympiadSolver
# ─────────────────────────────────────────────────────────────────────────────

class OlympiadSolver:

    def __init__(
        self,
        cfg:         SolverConfig = None,
        load_models: bool         = True,
        port:        int          = 8000,
    ):
        self.cfg  = cfg or SolverConfig()
        self.port = port
        self.notebook_start_time = time.time()
        self.problems_remaining  = 50

        print("Loading knowledge database...")
        self.db = KnowledgeDB(db_dir=self.cfg.db_dir)
        if not self.db.is_built():
            print("  WARNING: Knowledge DB not built. Run notebook cell 2 first.")

        self.type_classifier = None
        self.verify_scorer   = None

        if load_models:
            try:
                ckpt = os.path.join(self.cfg.ckpt_dir, "answer_type_classifier")
                self.type_classifier = AnswerTypeInference(ckpt)
                print("  AnswerTypeClassifier: loaded")
            except Exception as e:
                print(f"  AnswerTypeClassifier: skipped ({e})")

            try:
                ckpt = os.path.join(self.cfg.ckpt_dir, "verify_scorer")
                self.verify_scorer = VerifyScorerInference(ckpt)
                print("  VerifyScorer: loaded")
            except Exception as e:
                print(f"  VerifyScorer: skipped ({e})")

        print("Starting vLLM server...")
        self._start_vllm_server()

        print(f"Starting {self.cfg.workers} sandboxes...")
        self.sandbox_pool = queue.Queue()
        self._init_sandboxes()
        print("Ready.\n")

    def _start_vllm_server(self):
        import sys, subprocess
        cfg = self.cfg

        model_dir = cfg.model_path
        if os.path.isdir(model_dir):
            files = [str(p) for p in Path(model_dir).rglob("*.safetensors")]
            if not files:
                files = [str(p) for p in Path(model_dir).rglob("*.bin")]
            total_gb = sum(os.path.getsize(f) for f in files) / 1e9
            print(f"  Pre-loading {len(files)} weight files ({total_gb:.1f} GB)...")
            from concurrent.futures import ThreadPoolExecutor
            def _read(p):
                with open(p, "rb") as f:
                    while f.read(1 << 20):
                        pass
            with ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(_read, files))

        _gen_cfg_dir = "/kaggle/working/gen_cfg_override"
        os.makedirs(_gen_cfg_dir, exist_ok=True)
        with open(os.path.join(_gen_cfg_dir, "generation_config.json"), "w") as _f:
            json.dump({"transformers_version": "4.40.0"}, _f)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  cfg.model_path,
            "--served-model-name",      cfg.served_model_name,
            "--tensor-parallel-size",   "1",
            "--max-num-seqs",           str(cfg.batch_size),
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
            "--host",                   "0.0.0.0",
            "--port",                   str(self.port),
            "--dtype",                  cfg.dtype,
            "--kv-cache-dtype",         cfg.kv_cache_dtype,
            "--max-model-len",          str(cfg.context_tokens),
            "--stream-interval",        str(cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
            "--generation-config",      _gen_cfg_dir,
        ]

        self._log_file    = open("vllm_server.log", "w")
        self._server_proc = subprocess.Popen(
            cmd, stdout=self._log_file, stderr=subprocess.STDOUT,
            start_new_session=True)

        from openai import OpenAI
        self.client = OpenAI(base_url=f"http://localhost:{self.port}/v1",
                             api_key="placeholder")

        print("  Waiting for vLLM server...", end="", flush=True)
        for i in range(cfg.server_timeout):
            if self._server_proc.poll() is not None:
                with open("vllm_server.log") as lf:
                    raise RuntimeError(f"vLLM server died:\n{lf.read()}")
            try:
                self.client.models.list()
                print(f" ready in {i}s")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("vLLM server did not start in time.")

    def _init_sandboxes(self):
        from llm import MathSandbox
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(MathSandbox, self.cfg.jupyter_timeout)
                    for _ in range(self.cfg.workers)]
            for fut in as_completed(futs):
                try:
                    self.sandbox_pool.put(fut.result())
                except Exception as e:
                    print(f"  Sandbox init warning: {e}")

    def _get_forced_type(self, problem: str) -> Optional[str]:
        if self.type_classifier:
            try:
                return self.type_classifier.predict(problem)
            except Exception:
                pass
        return None

    def solve_problem(self, problem: str):
        print(f"\n{'─'*60}")
        print(f"Problem: {problem[:120]}{'...' if len(problem)>120 else ''}")

        elapsed   = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed
        reserved  = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        budget    = min(time_left - reserved, self.cfg.high_problem_timeout)
        budget    = max(budget, self.cfg.base_problem_timeout)
        deadline  = time.time() + budget
        print(f"Budget: {budget:.0f}s")

        forced_type = self._get_forced_type(problem)
        if forced_type:
            print(f"Predicted answer type: {forced_type}")

        stop_event = threading.Event()
        executor   = ThreadPoolExecutor(max_workers=self.cfg.attempts)
        results: list[TypedAnswer] = []

        def _attempt(idx):
            sandbox = None
            try:
                sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
                dispatcher = ToolDispatcher(db=self.db, sandbox=sandbox)
                return run_attempt(
                    problem, self.client, dispatcher, self.cfg,
                    attempt_idx=idx,
                    stop_event=stop_event,
                    deadline=deadline,
                    forced_type=forced_type,
                )
            except queue.Empty:
                return None
            except Exception as e:
                print(f"  [attempt {idx}] uncaught: {e}")
                return None
            finally:
                if sandbox is not None:
                    try:
                        sandbox.reset()
                    except Exception:
                        pass
                    self.sandbox_pool.put(sandbox)

        futures = {executor.submit(_attempt, i): i
                   for i in range(self.cfg.attempts)}

        try:
            for fut in as_completed(futures, timeout=budget):
                if stop_event.is_set():
                    break
                try:
                    ans = fut.result()
                    if ans is not None:
                        results.append(ans)
                        print(f"  attempt done: {ans}")
                        groups = []
                        for a in results:
                            placed = False
                            for g in groups:
                                if answers_match(a, g[0]):
                                    g.append(a)
                                    placed = True
                                    break
                            if not placed:
                                groups.append([a])
                        if groups and max(len(g) for g in groups) >= self.cfg.early_stop_votes:
                            stop_event.set()
                            break
                except Exception as e:
                    print(f"  attempt exception: {e}")
        finally:
            stop_event.set()
            executor.shutdown(wait=False, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)

        if not results:
            print("No answers found. Returning fallback 0.")
            return 0

        best = select_best_answer(results, min_votes=2)
        if best is None:
            best = max(results, key=lambda x: x.confidence)

        print(f"Final answer: {best.value}  (type={best.answer_type}, "
              f"conf={best.confidence:.2f})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best.value

    def __del__(self):
        with contextlib.suppress(Exception):
            self._server_proc.terminate()
            self._server_proc.wait()
        with contextlib.suppress(Exception):
            self._log_file.close()
        while not self.sandbox_pool.empty():
            with contextlib.suppress(Exception):
                self.sandbox_pool.get_nowait().close()


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle competition entry point
# ─────────────────────────────────────────────────────────────────────────────

_solver: Optional[OlympiadSolver] = None


def get_solver() -> OlympiadSolver:
    global _solver
    if _solver is None:
        _solver = OlympiadSolver()
    return _solver


def predict(id_, question, answer=None):
    import polars as pl
    id_value      = id_.item(0)
    question_text = question.item(0)
    solver = get_solver()
    result = solver.solve_problem(question_text)
    return pl.DataFrame({"id": [id_value], "answer": [result]})
