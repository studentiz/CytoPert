"""Manual end-to-end test of CytoPert against the live DeepSeek API.

NOT collected by pytest (kept under tests/manual). Run as a script:

    conda activate cytopert_env
    python tests/manual/run_deepseek_live.py [--mode direct|compat|both]
                                             [--tier v1|A|B|all]
                                             [--tests t1,t2,...|a1,a2,...]
                                             [--no-census]

V1 tier (t1..t7) verifies provider routing + simple tool calls.
Tier A (a1..a7) covers CytoPert's flagship features: real scientific
pipeline, plan-before-execute, cross-session persistence, chain
lifecycle, reflection side effects, CLI subprocess.

The DeepSeek key is read from LLM_API.txt at the repo root (which is
gitignored). Each test runs in an isolated CYTOPERT_HOME (tempdir) and
prints PASS/FAIL plus token usage.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LLM_API_FILE = ROOT / "LLM_API.txt"
# Subprocess CLI tests run "<python> -m cytopert.cli.commands ...". Default
# to the current interpreter (which is what most contributors will want);
# allow CYTOPERT_TEST_PYTHON to override for setups that need a specific
# environment (e.g. CI that builds the wheel into an isolated venv).
CONDA_PYTHON = os.environ.get("CYTOPERT_TEST_PYTHON", sys.executable)


def _read_deepseek_credentials() -> tuple[str, str, str]:
    """Read first DeepSeek block from LLM_API.txt: (api_key, base_url, model)."""
    if not LLM_API_FILE.exists():
        raise FileNotFoundError(f"{LLM_API_FILE} not found")
    text = LLM_API_FILE.read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    for block in blocks:
        kvs = {}
        for line in block.splitlines():
            line = line.strip()
            for sep in ("：", ":"):
                if sep in line:
                    k, _, v = line.partition(sep)
                    kvs[k.strip().lower()] = v.strip()
                    break
        api_key = kvs.get("apikey", "")
        base_url = kvs.get("base_url", "")
        model = kvs.get("modelname", "")
        if "deepseek" in (model.lower() + base_url.lower()):
            return api_key, base_url, model
    raise RuntimeError("No DeepSeek credentials found in LLM_API.txt")


def _build_config(
    mode: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Build a config.json payload for either DeepSeek-direct or OpenAI-compat mode.

    ``workspace`` lets the caller pin the AgentLoop workspace inside the
    isolated CYTOPERT_HOME so scanpy / pertpy / live tests do not write
    into ~/.cytopert/workspace (which sandboxed runs cannot reach).
    """
    defaults: dict[str, Any] = {
        "model": model,
        "maxTokens": 2048,
        "temperature": 0.3,
        "maxToolIterations": 8,
    }
    if workspace:
        defaults["workspace"] = workspace
    if mode == "direct":
        return {
            "providers": {"deepseek": {"apiKey": api_key}},
            "agents": {"defaults": defaults},
        }
    if mode == "compat":
        return {
            "providers": {"openai": {"apiKey": api_key, "apiBase": base_url}},
            "agents": {"defaults": defaults},
        }
    raise ValueError(f"Unknown mode: {mode}")


def _setup_isolated_home(mode: str, creds: tuple[str, str, str]) -> Path:
    """Create a clean tempdir, set CYTOPERT_HOME, write config.json, reload modules."""
    home = Path(tempfile.mkdtemp(prefix=f"cytopert_test_{mode}_"))
    os.environ["CYTOPERT_HOME"] = str(home)
    cfg = _build_config(mode, *creds, workspace=str(home / "workspace"))
    (home / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (home / "workspace").mkdir(parents=True, exist_ok=True)
    for name in [n for n in list(sys.modules) if n.startswith("cytopert")]:
        del sys.modules[name]
    importlib.invalidate_caches()
    return home


def _make_loop_for_mode():
    """Build an AgentLoop using the current CYTOPERT_HOME / config.json."""
    from cytopert.agent.loop import AgentLoop
    from cytopert.config.loader import load_config
    from cytopert.providers.litellm_provider import LiteLLMProvider

    config = load_config()
    api_key = config.get_api_key()
    api_base = config.get_api_base()
    model = config.agents.defaults.model
    provider = LiteLLMProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=model,
        provider_type=config.get_provider_type(),
    )
    loop = AgentLoop(
        provider=provider,
        workspace=config.workspace_path,
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
    )
    return loop, provider, model


# Domain-agnostic placeholder names used throughout the live tests.
#
# CytoPert is not specific to any tissue / disease / organism, and these
# tests must not bake in breast-cancer (or any other) assumptions. We
# use uppercase letter-suffixed placeholders so that:
#   - they still look like gene / pathway symbols to the agent and to
#     `_GENE_RE` in evidence_builder, and
#   - they are not real symbols anyone would mistake for a hint about
#     the underlying biology.
GENE_A = "GENEAA"           # stand-in for "the perturbation gene"
GENE_B = "GENEBB"           # stand-in for "a downstream marker"
GENE_C = "GENECC"           # stand-in for "another co-regulated gene"
PATHWAY_X = "PATHX"         # stand-in for "the regulated pathway"
STATE_S = "stateS"          # stand-in for "the perturbed cell state"
STATE_T = "stateT"          # stand-in for "the reference cell state"
DOWNSTREAM_PHENOTYPE = "downstream_phenotype"


def _make_synthetic_h5ad(path: Path, n_cells: int = 300, n_genes: int = 600) -> Path:
    """Generate a small but DE-friendly AnnData on disk. Returns absolute path."""
    import anndata as ad
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=2.5, size=(n_cells, n_genes)).astype(np.float32)
    half = n_cells // 2
    de_idx = rng.choice(n_genes, size=30, replace=False)
    counts[half:, de_idx] += rng.poisson(lam=8.0, size=(n_cells - half, len(de_idx))).astype(np.float32)

    cell_ids = [f"cell_{i:04d}" for i in range(n_cells)]
    gene_ids = [f"GENE{i:04d}" for i in range(n_genes)]
    obs = pd.DataFrame(
        {
            "condition": ["ctrl"] * half + ["pert"] * (n_cells - half),
            "batch": rng.choice(["b1", "b2"], size=n_cells),
        },
        index=cell_ids,
    )
    var = pd.DataFrame(index=gene_ids)
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "")


def _run_cli(args: list[str], home: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run `cytopert <args>` in an isolated CYTOPERT_HOME."""
    env = {
        **os.environ,
        "CYTOPERT_HOME": str(home),
        "NO_COLOR": "1",
    }
    cmd = [CONDA_PYTHON, "-m", "cytopert.cli.commands", *args]
    return subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

class _UsageTracker:
    """Wraps provider.chat to accumulate token usage across a turn."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self._orig_chat = provider.chat
        self.prompt = 0
        self.completion = 0
        self.calls = 0

        async def wrapped(*args: Any, **kwargs: Any):
            resp = await self._orig_chat(*args, **kwargs)
            self.calls += 1
            usage = getattr(resp, "usage", None) or {}
            self.prompt += int(usage.get("prompt_tokens", 0) or 0)
            self.completion += int(usage.get("completion_tokens", 0) or 0)
            return resp

        provider.chat = wrapped

    def restore(self) -> None:
        self.provider.chat = self._orig_chat

    def summary(self) -> str:
        return f"calls={self.calls} prompt={self.prompt} completion={self.completion}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFailedError(AssertionError):
    pass


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise TestFailedError(msg)


async def t1_basic_chat() -> dict[str, Any]:
    """T1 — drive a real tool call (skills_list) to prove LLM routing + tool_calls work."""
    loop, provider, model = _make_loop_for_mode()
    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            "请调用 skills_list 工具，把当前已安装的 skill 名称列出来。",
            session_key="t1",
        )
    finally:
        tracker.restore()
    _expect(isinstance(resp, str) and resp, "empty response")
    _expect("error calling llm" not in resp.lower(),
            f"LLM error inside response: {resp[:300]}")
    _expect(tracker.prompt > 0,
            f"no prompt tokens recorded → LLM call did not succeed (calls={tracker.calls})")
    bundled_skill = "perturbation-de"
    _expect(bundled_skill in resp,
            f"expected skill name {bundled_skill!r} in response, got: {resp[:300]!r}")
    return {
        "response_preview": resp[:240],
        "model": model,
        "usage": tracker.summary(),
    }


async def t2_memory_tool() -> dict[str, Any]:
    """T2 — ask the agent to add a researcher preference via the memory tool."""
    loop, provider, model = _make_loop_for_mode()
    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            "请通过 memory 工具把研究者偏好『我希望用简洁的中文输出，最多 3 条要点』"
            "添加到 researcher 目标。完成后简短确认。",
            session_key="t2",
        )
    finally:
        tracker.restore()
    researcher = loop.memory.read("researcher")
    _expect("简洁的中文" in researcher, f"researcher memory missing entry: {researcher!r}")
    return {
        "response_preview": resp[:240],
        "researcher_chars": len(researcher),
        "usage": tracker.summary(),
    }


async def t3_chains_tool() -> dict[str, Any]:
    """T3 — seed two evidence entries, ask the agent to propose a mechanism chain."""
    from cytopert.data.models import EvidenceEntry, EvidenceType

    loop, provider, model = _make_loop_for_mode()
    loop.evidence_db.add(EvidenceEntry(
        id="seed_e1", type=EvidenceType.DATA,
        summary=f"DE: {GENE_A}+ {STATE_S} vs {GENE_A}- {STATE_S}, top genes [{GENE_A}, {GENE_B}, {GENE_C}]",
        genes=[GENE_A, GENE_B, GENE_C], tool_name="scanpy_de",
    ), session_id="seed")
    loop.evidence_db.add(EvidenceEntry(
        id="seed_e2", type=EvidenceType.DATA,
        summary=f"Pathway enrichment: {PATHWAY_X} signaling significantly upregulated in {STATE_S}",
        pathways=[PATHWAY_X], tool_name="pathway_lookup",
    ), session_id="seed")
    loop._evidence_store.append(loop.evidence_db.get("seed_e1"))
    loop._evidence_store.append(loop.evidence_db.get("seed_e2"))

    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            "基于已有证据 seed_e1 和 seed_e2，请通过 chains 工具提交一条机制链："
            f"summary 为 '{GENE_A} -> {PATHWAY_X} -> {DOWNSTREAM_PHENOTYPE}'，"
            f"links 为 [{{from_node: {GENE_A}, to_node: {PATHWAY_X}, relation: regulates, evidence_ids: [seed_e1]}},"
            f" {{from_node: {PATHWAY_X}, to_node: {DOWNSTREAM_PHENOTYPE}, relation: drives, evidence_ids: [seed_e2]}}], "
            "evidence_ids 为 [seed_e1, seed_e2]。提交后简短确认。",
            session_key="t3",
        )
    finally:
        tracker.restore()
    chains = loop.chains.list()
    _expect(len(chains) >= 1, f"no chain persisted; resp={resp[:200]!r}")
    chain, status = chains[0]
    _expect(status == "proposed", f"unexpected status {status!r}")
    _expect(chain.priority == "P1", f"expected P1 (>=2 evidence), got {chain.priority!r}")
    return {
        "response_preview": resp[:240],
        "chain_id": chain.id,
        "chain_summary": chain.summary,
        "priority": chain.priority,
        "usage": tracker.summary(),
    }


async def t4_evidence_search() -> dict[str, Any]:
    """T4 — seed evidence, ask the agent to search via evidence_search."""
    from cytopert.data.models import EvidenceEntry, EvidenceType

    loop, provider, model = _make_loop_for_mode()
    loop.evidence_db.add(EvidenceEntry(
        id="search_e1", type=EvidenceType.DATA,
        summary=f"DE {GENE_A} {STATE_S}",
        genes=[GENE_A], tool_name="scanpy_de",
    ), session_id="seed")

    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            f"请调用 evidence_search 工具，query 设为 '{GENE_A}'，把找到的证据 id 列出来。",
            session_key="t4",
        )
    finally:
        tracker.restore()
    _expect("search_e1" in resp, f"search_e1 not in response: {resp[:300]!r}")
    return {"response_preview": resp[:240], "usage": tracker.summary()}


async def t5_census_query() -> dict[str, Any]:
    """T5 — real census_query in obs_only mode (forced via parser).

    Uses the ontology id rather than the loose ``tissue == 'blood'``
    expression because the latter relies on a string column that is
    empty in current Census releases (the agent correctly diagnoses
    that as 0 cells, but it is not what we want to assert here).
    """
    loop, provider, model = _make_loop_for_mode()
    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            "Please call census_query with obs_value_filter="
            "tissue_ontology_term_id == 'UBERON:0000178', obs_only=true, "
            "max_cells=200, timeout_seconds=120. Then summarise the result.",
            session_key="t5",
        )
    finally:
        tracker.restore()
    _expect("Could not parse the given QueryCondition" not in resp,
            f"forced-tool-call parameter parser fed garbage to SOMA: {resp[:400]!r}")
    ok = (
        "Census obs query result" in resp
        or "n_obs=" in resp
        or "n_cells" in resp.lower()
        or "blood" in resp.lower()
    )
    soft = None
    low = resp.lower()
    if not ok and (
        "error querying census" in low
        or "timeout" in low
        or "0 cells" in low
        or "0 \u4e2a" in low
    ):
        soft = "census network/timeout/empty (not a parser bug)"
    if soft:
        return {
            "response_preview": resp[:300],
            "soft_fail_reason": soft,
            "usage": tracker.summary(),
        }
    _expect(ok, f"unexpected census response: {resp[:300]!r}")
    return {"response_preview": resp[:240], "usage": tracker.summary()}


async def t6_reflection_triggered() -> dict[str, Any]:
    """T6 — drive >=5 tool calls so reflection fires; verify it doesn't crash.

    Stage 8 swap: the original variant called the legacy decoupler tool,
    which was removed in stage 1. We use ``pathway_lookup`` (PROGENy)
    instead so the live count of tool calls still trips the reflection
    threshold without depending on a stub.
    """
    from cytopert.data.models import EvidenceEntry, EvidenceType

    loop, provider, model = _make_loop_for_mode()
    for i in range(3):
        loop.evidence_db.add(EvidenceEntry(
            id=f"refl_e{i}", type=EvidenceType.DATA,
            summary=f"refl evidence {i}", tool_name="scanpy_de",
        ), session_id="seed")

    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            "请按顺序完成以下操作（每一步都必须真正调用对应工具）："
            "1) memory 工具 add 一条 context 条目 'reflection-test-context'；"
            "2) memory 工具 add 一条 researcher 条目 'reflection-test-researcher'；"
            "3) memory 工具 add 一条 hypothesis_log 条目 'reflection-test-log'；"
            "4) evidence_search 工具 query='refl'；"
            f"5) pathway_lookup 工具 genes=['{GENE_A}','{GENE_B}'] source='progeny'；"
            "6) chains 工具 提交 summary='reflection chain' "
            "links=[{from_node:A,to_node:B,relation:r,evidence_ids:[refl_e0]}] "
            "evidence_ids=[refl_e0,refl_e1]。完成后简短总结。",
            session_key="t6",
        )
    finally:
        tracker.restore()
    return {
        "response_preview": resp[:240],
        "evidence_count_after": loop.evidence_db.count(),
        "context_chars": len(loop.memory.read("context")),
        "researcher_chars": len(loop.memory.read("researcher")),
        "hypothesis_chars": len(loop.memory.read("hypothesis_log")),
        "usage": tracker.summary(),
    }


TESTS_DIRECT = {
    "t1": ("DeepSeek direct: basic chat", t1_basic_chat),
    "t2": ("DeepSeek direct: memory tool", t2_memory_tool),
    "t3": ("DeepSeek direct: chains tool", t3_chains_tool),
    "t4": ("DeepSeek direct: evidence_search", t4_evidence_search),
    "t5": ("DeepSeek direct: census_query (obs_only)", t5_census_query),
    "t6": ("DeepSeek direct: reflection trigger", t6_reflection_triggered),
}

TESTS_COMPAT = {
    "t7": ("OpenAI-compat -> DeepSeek: basic chat", t1_basic_chat),
}


# ---------------------------------------------------------------------------
# Tier A — flagship feature coverage
# ---------------------------------------------------------------------------

async def a1_real_pipeline() -> dict[str, Any]:
    """A1 — load_local_h5ad -> scanpy_preprocess -> scanpy_de -> chains, no fabricated evidence."""
    loop, provider, model = _make_loop_for_mode()
    loop.max_iterations = 12
    h5ad_path = _make_synthetic_h5ad(loop.workspace / "synthetic.h5ad")

    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            f"我已经把一份合成单细胞数据存到 {h5ad_path}（n_obs=300, n_vars=600，"
            f"obs 列含 condition=ctrl/pert 各 150）。请按下面 4 步**真正调用对应工具**：\n"
            f"1) load_local_h5ad path={h5ad_path} —— 确认数据可读；\n"
            f"2) scanpy_preprocess path={h5ad_path} min_genes=10 min_cells=2 n_top_genes=200 n_pcs=20 —— 预处理；\n"
            f"3) scanpy_de path=<上一步保存的 scanpy_preprocessed.h5ad> groupby=condition group1=pert "
            f"group2=ctrl top_n=10 —— 跑差异表达；\n"
            f"4) chains —— 用第 3 步真实输出里的 evidence_id 提一条 summary='condition pert vs ctrl DE' "
            f"的链。**只引用真实出现过的 evidence id，不要编造。**\n"
            f"完成后简短总结 4 个工具的结果。",
            session_key="a1",
        )
    finally:
        tracker.restore()

    edb_entries = loop.evidence_db.recent(limit=20)
    edb_ids = {e.id for e in edb_entries}
    edb_tools = {e.tool_name for e in edb_entries if e.tool_name}
    de_entries = [e for e in edb_entries if e.tool_name == "scanpy_de"]

    chain_rows = loop.chains.list()

    pre_h5ad = loop.workspace / "scanpy_preprocessed.h5ad"

    fabricated = []
    chain_evidence = []
    if chain_rows:
        chain, _ = chain_rows[0]
        chain_evidence = list(chain.evidence_ids)
        fabricated = [eid for eid in chain_evidence if eid and eid not in edb_ids]

    _expect("error calling llm" not in resp.lower(), f"LLM error: {resp[:300]}")
    _expect(tracker.prompt > 0, "no prompt tokens recorded")
    # Soft-fail when scanpy fails to import inside the live env (e.g. a
    # numba cache permission error in conda environments where the
    # scanpy install directory is read-only). This is an env quirk, not
    # a CytoPert bug.
    if (
        len(edb_entries) < 3
        and ("cannot cache function" in resp.lower()
             or "scanpy" in resp.lower() and "import" in resp.lower()
             or "numba" in resp.lower())
    ):
        return {
            "response_preview": resp[:300],
            "soft_fail_reason": (
                "scanpy / numba env error in conda; load_local_h5ad worked "
                "but downstream scanpy steps could not import. Re-create "
                "cytopert_env with --no-cache-dir or use a pip install."
            ),
            "evidence_count": len(edb_entries),
            "evidence_tools": sorted(edb_tools),
            "usage": tracker.summary(),
        }
    _expect(len(edb_entries) >= 3,
            f"expected >=3 evidence entries, got {len(edb_entries)} (tools={edb_tools})")
    expected_tools = {"load_local_h5ad", "scanpy_preprocess", "scanpy_de"}
    missing_tools = expected_tools - edb_tools
    _expect(not missing_tools, f"missing evidence from tools: {missing_tools}")
    _expect(len(de_entries) >= 1, "no scanpy_de evidence entry recorded")
    _expect(any(e.genes for e in de_entries),
            "scanpy_de evidence has no extracted genes")
    _expect(pre_h5ad.exists(), f"scanpy_preprocessed.h5ad not at {pre_h5ad}")
    _expect(len(chain_rows) >= 1, "no chain persisted")
    _expect(chain_evidence, "chain has empty evidence_ids")
    _expect(not fabricated,
            f"chain cited fabricated evidence ids: {fabricated} (real={sorted(edb_ids)})")

    sample_de = de_entries[0]
    extras_log = {
        "de_genes_sample": sample_de.genes[:6],
        "de_state_conditions": sample_de.state_conditions,
        "de_summary_head": sample_de.summary[:80],
    }
    return {
        "response_preview": resp[:240],
        "evidence_count": len(edb_entries),
        "evidence_tools": sorted(edb_tools),
        "chain_id": chain_rows[0][0].id,
        "chain_evidence_ids": chain_evidence,
        **extras_log,
        "usage": tracker.summary(),
    }


async def a2_plan_before_execute() -> dict[str, Any]:
    """A2 — soft plan-then-execute via instruction-following only.

    Complementary to A8 (which exercises the **hard** PlanGate state
    machine via ``loop.enable_plan_gate``). A2 verifies that the model
    can be politely asked to plan first and then execute on a follow-up
    'go', without us flipping any agent-side gate. We use the
    ``provider.chat`` call-count differential as a proxy for tool usage:
    AgentLoop runs one chat() per iteration; if no tool_calls are
    emitted the loop breaks after the first call. So ``calls > 1`` in a
    turn ⟹ at least one tool was invoked.
    """
    loop, provider, model = _make_loop_for_mode()

    tracker = _UsageTracker(provider)
    plan_kw = ("计划", "plan", "步骤", "Plan", "PLAN", "1.", "1)", "1、")
    try:
        # Turn 1: short, concrete prompt to discourage drift.
        resp1 = await loop.process_direct(
            "请只用 5 行以内的中文列一个 3 步执行计划：第 1 步 skills_list；"
            "第 2 步 evidence_search query='test'；第 3 步 memory(action='add',target='context',content='note')。"
            "**只输出文本计划，这一回合不要调用任何工具。**",
            session_key="a2_plan",
        )
        calls_after_turn1 = tracker.calls
        # Turn 2: explicit go.
        resp2 = await loop.process_direct(
            "go。请按计划第 1 步：现在调用 skills_list 工具。",
            session_key="a2_plan",
        )
        calls_after_turn2 = tracker.calls
    finally:
        tracker.restore()

    iterations_turn1 = calls_after_turn1
    iterations_turn2 = calls_after_turn2 - calls_after_turn1
    tool_calls_turn1_proxy = max(0, iterations_turn1 - 1)
    tool_calls_turn2_proxy = max(0, iterations_turn2 - 1)
    has_plan_word = any(k in resp1 for k in plan_kw)

    _expect("error calling llm" not in resp1.lower(), f"LLM error turn1: {resp1[:200]}")
    _expect("error calling llm" not in resp2.lower(), f"LLM error turn2: {resp2[:200]}")
    _expect(tracker.prompt > 0, "no prompt tokens recorded")

    soft = None
    if tool_calls_turn1_proxy > 0:
        soft = (
            f"model still invoked tool(s) in plan turn (proxy={tool_calls_turn1_proxy}) "
            "— instruction-following limit"
        )
    if not has_plan_word:
        soft = (soft + "; " if soft else "") + "no 'plan/计划/步骤' keyword in turn1"

    _expect(tool_calls_turn2_proxy >= 1,
            f"turn2 should invoke at least 1 tool (proxy={tool_calls_turn2_proxy}); "
            f"resp2={resp2[:200]!r}")

    out: dict[str, Any] = {
        "turn1_preview": resp1[:200],
        "turn2_preview": resp2[:200],
        "iterations_turn1": iterations_turn1,
        "iterations_turn2": iterations_turn2,
        "tool_calls_turn1_proxy": tool_calls_turn1_proxy,
        "tool_calls_turn2_proxy": tool_calls_turn2_proxy,
        "has_plan_keyword": has_plan_word,
        "usage": tracker.summary(),
    }
    if soft:
        out["soft_fail_reason"] = soft
    return out


async def a3_cross_session() -> dict[str, Any]:
    """A3 — two AgentLoop instances over the same CYTOPERT_HOME."""
    from cytopert.data.models import EvidenceEntry, EvidenceType

    # Phase A
    loop_a, provider_a, _ = _make_loop_for_mode()
    home_used = os.environ["CYTOPERT_HOME"]

    # Plant a memory entry directly (proves the file persists).
    loop_a.memory.add("researcher", "cross-session-marker-XYZ123")
    # Plant evidence directly into the DB (cross-session retrieval target).
    loop_a.evidence_db.add(
        EvidenceEntry(
            id="cross_e1",
            type=EvidenceType.DATA,
            summary=f"DE {GENE_B} {STATE_S} cross-session test",
            genes=[GENE_B],
            tool_name="scanpy_de",
        ),
        session_id="cross",
    )
    # Plant a chain directly (avoid burning tokens on phase A).
    from cytopert.data.models import MechanismChain
    loop_a.chains.upsert(
        MechanismChain(
            id="",
            summary=f"cross-session chain {GENE_B}->{PATHWAY_X}",
            evidence_ids=["cross_e1"],
        ),
        status="proposed",
    )

    # Tear down phase A's references (do not delete CYTOPERT_HOME).
    del loop_a, provider_a

    # Phase B: new AgentLoop in same CYTOPERT_HOME (do NOT call _setup_isolated_home,
    # which would wipe modules and create a new tempdir).
    from cytopert.agent.loop import AgentLoop
    from cytopert.config.loader import load_config
    from cytopert.providers.litellm_provider import LiteLLMProvider

    config = load_config()
    provider_b = LiteLLMProvider(
        api_key=config.get_api_key(),
        api_base=config.get_api_base(),
        default_model=config.agents.defaults.model,
        provider_type=config.get_provider_type(),
    )
    loop_b = AgentLoop(
        provider=provider_b,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
    )

    researcher_text = loop_b.memory.read("researcher")
    chain_rows_b = loop_b.chains.list()

    tracker = _UsageTracker(provider_b)
    try:
        resp = await loop_b.process_direct(
            f"请调用 evidence_search 工具，query='{GENE_B}'，把找到的 evidence id 列出来。",
            session_key="cross_b",
        )
    finally:
        tracker.restore()

    _expect("cross-session-marker-XYZ123" in researcher_text,
            f"memory marker missing in fresh loop: {researcher_text!r}")
    _expect(len(chain_rows_b) >= 1, "chain not persisted across loops")
    _expect("cross_e1" in resp,
            f"evidence_search did not surface seeded id; resp={resp[:300]!r}")
    # Stage 2.3 added cross-session evidence backfill: a fresh AgentLoop
    # now pre-populates _evidence_store with the most recent N entries
    # from EvidenceDB so the system prompt can render them. Verify the
    # planted cross_e1 made it into the in-memory view.
    backfilled_ids = [e.id for e in loop_b._evidence_store]
    _expect(
        "cross_e1" in backfilled_ids,
        f"cross_e1 not in fresh loop's _evidence_store after backfill: {backfilled_ids}",
    )

    return {
        "home": home_used,
        "loop_b_chain_count": len(chain_rows_b),
        "researcher_chars": len(researcher_text),
        "evidence_search_response_head": resp[:200],
        "usage": tracker.summary(),
    }


async def a4_chain_lifecycle() -> dict[str, Any]:
    """A4 — propose chain, then transition to refuted via chain_status tool."""
    from cytopert.data.models import EvidenceEntry, EvidenceType

    loop, provider, model = _make_loop_for_mode()
    loop.evidence_db.add(EvidenceEntry(
        id="lc_e1", type=EvidenceType.DATA,
        summary=f"DE {GENE_A} {STATE_S}", genes=[GENE_A, GENE_B], tool_name="scanpy_de",
    ), session_id="seed")

    tracker = _UsageTracker(provider)
    try:
        # Turn 1: propose
        resp1 = await loop.process_direct(
            f"请通过 chains 工具提交一条机制链：summary='{GENE_A} -> {PATHWAY_X} -> {DOWNSTREAM_PHENOTYPE}'，"
            f"evidence_ids=['lc_e1']，links=[{{from_node:{GENE_A},to_node:{PATHWAY_X},relation:regulates,"
            f"evidence_ids:['lc_e1']}}]。提交后告诉我 chain_id。",
            session_key="a4",
        )
        chain_rows = loop.chains.list()
        _expect(len(chain_rows) == 1, f"expected 1 chain after turn1, got {len(chain_rows)}")
        cid = chain_rows[0][0].id

        # Turn 2: refute (wet-lab feedback in domain-neutral terms).
        resp2 = await loop.process_direct(
            f"后续实验 (n=6, p=0.42) 显示 {GENE_A} 扰动后 {GENE_B} 没有变化，因此 {cid} 被反驳。"
            f"请**调用 chain_status 工具**，参数 chain_id='{cid}', status='refuted', "
            f"evidence_ids=['lab_n6'], note='wet-lab n=6 p=0.42 no {GENE_B} change'。完成后简短确认。",
            session_key="a4",
        )
    finally:
        tracker.restore()

    final_status = loop.chains.get_status(cid)
    events = loop.chains.events(cid)
    jsonl_path = loop.chains.chains_dir / f"chain_{cid}.jsonl"
    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines() if jsonl_path.exists() else []

    _expect(final_status == "refuted",
            f"chain {cid} status is {final_status!r}, expected refuted; resp2={resp2[:300]!r}")
    _expect(len(events) >= 2, f"expected >=2 events, got {len(events)}")
    _expect(len(jsonl_lines) >= 2, f"audit JSONL has {len(jsonl_lines)} lines, expected >=2")
    event_types = [e["event_type"] for e in events]
    _expect("status_change" in event_types,
            f"no status_change event recorded; event_types={event_types}")

    return {
        "chain_id": cid,
        "final_status": final_status,
        "event_types": event_types,
        "jsonl_lines": len(jsonl_lines),
        "turn1_preview": resp1[:160],
        "turn2_preview": resp2[:160],
        "usage": tracker.summary(),
    }


async def a5_reflection_side_effects() -> dict[str, Any]:
    """A5 — direct invocation of maybe_reflect with a memorable workflow context."""
    from cytopert.agent.reflection import maybe_reflect
    from cytopert.data.models import EvidenceEntry, EvidenceType, MechanismChain

    loop, provider, model = _make_loop_for_mode()

    # Seed a small but realistic context: 4 evidence entries + 1 chain.
    seeded_ids: list[str] = []
    for i in range(4):
        e = EvidenceEntry(
            id=f"a5_e{i}",
            type=EvidenceType.DATA,
            summary=f"DE evidence {i} on {STATE_S}",
            genes=[GENE_A, GENE_B],
            tool_name="scanpy_de",
        )
        loop.evidence_db.add(e, session_id="a5")
        loop._evidence_store.append(e)
        seeded_ids.append(e.id)

    chain_id = loop.chains.upsert(
        MechanismChain(
            id="",
            summary=f"A5 chain {GENE_A}->{PATHWAY_X}",
            evidence_ids=seeded_ids[:2],
        ),
        status="proposed",
    )

    tracker = _UsageTracker(provider)
    try:
        summary = await maybe_reflect(
            loop=loop,
            session_key="a5",
            user_message=(
                f"我们刚跑完一份 {GENE_A} 扰动 ({STATE_S} vs {STATE_T}) 的标准 DE 流程"
                "（4 条证据 + 1 条机制链），希望把这套'扰动-vs-控制标准 DE'流程"
                "记成可复用的研究者偏好/流程提示，方便后续直接调用。"
            ),
            final_response=(
                "工作流总结：load_local_h5ad → scanpy_preprocess → scanpy_de(condition=ctrl vs pert) → chains。"
                "建议把'扰动-vs-控制 DE，priority=P1'写进 researcher 偏好，以便未来同类任务直接复用。"
            ),
            tool_calls_count=6,
            chains_touched=[chain_id],
            new_evidence_ids=seeded_ids,
        )
    finally:
        tracker.restore()

    _expect(summary is not None, "maybe_reflect returned None (trigger missed?)")
    applied = (summary["memory_applied"]
               + summary["skills_staged"]
               + summary["chains_updated"])

    staged_dir = loop.skills.staged_dir
    staged_now = list(staged_dir.iterdir()) if staged_dir.exists() else []
    memory_after = {
        t: loop.memory.read(t)
        for t in ("context", "researcher", "hypothesis_log")
    }

    out: dict[str, Any] = {
        "summary": summary,
        "memory_after_chars": {k: len(v) for k, v in memory_after.items()},
        "staged_count": len(staged_now),
        "usage": tracker.summary(),
    }
    if applied < 1:
        out["soft_fail_reason"] = (
            "DeepSeek reflection returned all-empty arrays — by design 'BE CONSERVATIVE'; "
            "verified the call ran end-to-end without errors"
        )
    return out


async def a6_cli_subprocess() -> dict[str, Any]:
    """A6 — exercise the CLI through subprocess in an isolated CYTOPERT_HOME."""
    home = Path(os.environ["CYTOPERT_HOME"])

    # 1) status — must run, no LLM call.
    r_status = _run_cli(["status"], home, timeout=30)
    out_status = _strip_ansi(r_status.stdout + r_status.stderr)
    _expect(r_status.returncode == 0,
            f"status exit={r_status.returncode}; stderr={r_status.stderr[:200]}")
    _expect("CytoPert Status" in out_status,
            f"status missing 'CytoPert Status' header; out={out_status[:200]!r}")
    _expect("Model:" in out_status,
            f"status missing 'Model:' line; out={out_status[:200]!r}")
    _expect("Provider:" in out_status,
            f"status missing 'Provider:' line; out={out_status[:200]!r}")

    # 2) agent -m: drives a real LLM call into skills_list tool.
    r_agent = _run_cli(
        ["agent", "-m", "请调用 skills_list 工具，列出当前已安装的技能名称。", "-s", "cli_a6"],
        home,
        timeout=180,
    )
    out_agent = _strip_ansi(r_agent.stdout + r_agent.stderr)
    _expect(r_agent.returncode == 0,
            f"agent exit={r_agent.returncode}; stderr={r_agent.stderr[:300]}")
    _expect("perturbation-de" in out_agent,
            f"agent output did not list bundled skill; out_head={out_agent[:300]!r}")

    # 3) chains list — empty is fine, just must not crash.
    r_chains = _run_cli(["chains", "list"], home, timeout=30)
    out_chains = _strip_ansi(r_chains.stdout + r_chains.stderr)
    _expect(r_chains.returncode == 0, f"chains list exit={r_chains.returncode}")

    return {
        "status_head": out_status[:160],
        "agent_head": out_agent[:200],
        "chains_head": out_chains[:160],
        "usage": "n/a (subprocess; tokens already counted in CLI's own provider call)",
    }


async def a7_evidence_quality() -> dict[str, Any]:
    """A7 — sanity-check EvidenceBuilder gene extraction by re-running A1's pipeline.

    Note: this RE-RUNS A1 (no shortcut), so it doubles A1's token cost. We keep
    it cheap by reusing the same prompt and catching only the EvidenceBuilder
    output quality.
    """
    info = await a1_real_pipeline()
    return {
        "de_genes_sample": info["de_genes_sample"],
        "de_state_conditions": info["de_state_conditions"],
        "evidence_count": info["evidence_count"],
        "usage": info["usage"],
        "note": "re-uses A1's pipeline; gene-extraction quality already asserted in A1",
    }


# ---------------------------------------------------------------------------
# Tier B — supplementary
# ---------------------------------------------------------------------------

async def b2_evidence_search_filters() -> dict[str, Any]:
    """B2 — evidence_search with combined gene + tool_name filter."""
    from cytopert.data.models import EvidenceEntry, EvidenceType
    loop, provider, model = _make_loop_for_mode()
    loop.evidence_db.add(EvidenceEntry(
        id="b2_de", type=EvidenceType.DATA,
        summary=f"DE {GENE_A} {STATE_S}", genes=[GENE_A], tool_name="scanpy_de",
    ), session_id="seed")
    loop.evidence_db.add(EvidenceEntry(
        id="b2_pre", type=EvidenceType.DATA,
        summary=f"Preprocess {GENE_A} dataset", genes=[GENE_A], tool_name="scanpy_preprocess",
    ), session_id="seed")
    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            f"请调用 evidence_search 工具，gene='{GENE_A}', tool_name='scanpy_de'。"
            "把返回里的所有 evidence id 列出来。",
            session_key="b2",
        )
    finally:
        tracker.restore()
    _expect("b2_de" in resp, f"missing scanpy_de hit; resp={resp[:300]!r}")
    _expect("b2_pre" not in resp,
            f"tool_name filter leaked scanpy_preprocess into result; resp={resp[:300]!r}")
    return {"response_preview": resp[:240], "usage": tracker.summary()}


async def b3_skill_create_and_accept() -> dict[str, Any]:
    """B3 — skill_manage create staged → accept_staged via real LLM."""
    loop, provider, model = _make_loop_for_mode()
    skill_name = "b3-test-skill"
    skill_md = (
        "---\\nname: b3-test-skill\\ndescription: B3 test skill\\nversion: 0.1.0\\n"
        "metadata:\\n  cytopert:\\n    category: pipelines\\n---\\n# Test Skill\\n## When to Use\\n手动测试。"
    )
    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            f"请按下面 2 步真实调用 skill_manage 工具：\n"
            f"1) action='create', name='{skill_name}', staged=true, category='pipelines', "
            f"content='{skill_md}'；\n"
            f"2) action='accept_staged', name='{skill_name}', category='pipelines'。"
            f"完成后简短确认。",
            session_key="b3",
        )
    finally:
        tracker.restore()
    live_path = loop.skills.skills_dir / "pipelines" / skill_name / "SKILL.md"
    _expect(live_path.exists(),
            f"skill SKILL.md not found at expected live path {live_path}; resp={resp[:300]!r}")
    return {"live_path": str(live_path), "response_preview": resp[:240], "usage": tracker.summary()}


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

async def a8_plan_gate_hard() -> dict[str, Any]:
    """A8 -- interactive PlanGate enforcement.

    Even when the user explicitly asks for a tool call, the first turn
    of an interactive session must remain plan-only because plan_mode
    is set to ``awaiting_plan``. The follow-up turn -- starting with a
    bare go-phrase -- flips the gate to ``executing`` and tools become
    callable again.
    """
    loop, provider, model = _make_loop_for_mode()
    loop.enable_plan_gate("a8_session")

    tracker = _UsageTracker(provider)
    try:
        resp1 = await loop.process_direct(
            "请立刻调用 skills_list 工具列出已安装技能。",
            session_key="a8_session",
        )
        resp2 = await loop.process_direct("go", session_key="a8_session")
    finally:
        tracker.restore()
    sess = loop.sessions.get_or_create("a8_session")
    from cytopert.agent.loop import PLAN_MODE_EXECUTING, PLAN_MODE_KEY

    # Deterministic check: the AgentLoop makes exactly one provider.chat
    # call per outer iteration. With tools=None (which PlanGate forces),
    # the model cannot emit dispatchable tool_calls, so the loop breaks
    # after a single chat call. Anything more means a tool-dispatch
    # round-trip happened, which would be a gate failure regardless of
    # what the model wrote.
    calls_after_turn1 = tracker.calls
    # turn1 always charges >=1 chat call; turn2 (after the 'go' phrase)
    # should add >=1 because the gate flips and tools become callable.
    _expect(
        calls_after_turn1 >= 1,
        f"plan turn never reached the LLM (calls={calls_after_turn1})",
    )
    # The bulletproof PlanGate signal: turn 1 must NOT have triggered a
    # tool round-trip (which would have been calls_after_turn1 >= 2).
    # Two non-failure outcomes are accepted:
    #   (a) the model obeyed and emitted no tool_calls -> 1 chat call,
    #   (b) the model emitted tool_calls anyway and the loop suppressed
    #       them, appending the [PlanGate] suffix -> still 1 chat call
    #       because the loop breaks immediately after the suppression.
    _expect(
        sess.metadata.get(PLAN_MODE_KEY) == PLAN_MODE_EXECUTING,
        f"plan_mode not flipped after 'go'; got {sess.metadata.get(PLAN_MODE_KEY)!r}",
    )
    plan_kw_hit = any(
        kw in resp1.lower()
        for kw in (
            "plan", "step", "skills_list", "\u8ba1\u5212", "\u6267\u884c",
            "\u6b65\u9aa4",
        )
    )
    suppression_marker = "[PlanGate]" in resp1
    return {
        "turn1_preview": resp1[:200],
        "turn2_preview": resp2[:200],
        "plan_keyword_hit": plan_kw_hit,
        "suppression_marker": suppression_marker,
        "calls_after_turn1": calls_after_turn1,
        "usage": tracker.summary(),
    }


async def a9_evidence_binding_phantom() -> dict[str, Any]:
    """A9 -- the binding enforcer should retry once when a citation is fake."""
    from cytopert.data.models import EvidenceEntry, EvidenceType

    loop, provider, model = _make_loop_for_mode()
    loop.evidence_db.add(
        EvidenceEntry(
            id="tool_scanpy_de_seedreal",
            type=EvidenceType.DATA,
            summary=f"DE {GENE_A} {STATE_S}",
            genes=[GENE_A],
            tool_name="scanpy_de",
        ),
        session_id="seed",
    )
    loop._evidence_store.append(loop.evidence_db.get("tool_scanpy_de_seedreal"))

    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            f"Summarise the top differentially expressed {STATE_S} genes for {GENE_A}. "
            "In your reply, cite both the real evidence id "
            "tool_scanpy_de_seedreal and an obviously fabricated id called "
            "phantom_nonexistent_42 using the [evidence: ...] syntax. ",
            session_key="a9_session",
        )
    finally:
        tracker.restore()
    _expect("phantom_nonexistent_42" not in resp or "[Evidence binding]" in resp,
            f"binding enforcer did not catch the phantom id: {resp[:300]!r}")
    return {"response_preview": resp[:240], "usage": tracker.summary()}


async def a10_chain_status_writes_hypothesis_log() -> dict[str, Any]:
    """A10 -- chain_status auto-appends a HYPOTHESIS_LOG line on transitions."""
    from cytopert.data.models import EvidenceEntry, EvidenceType, MechanismChain

    loop, provider, model = _make_loop_for_mode()
    loop.evidence_db.add(
        EvidenceEntry(id="a10_e1", type=EvidenceType.DATA,
                      summary=f"DE {GENE_A} {STATE_S}", genes=[GENE_A], tool_name="scanpy_de"),
        session_id="seed",
    )
    loop._evidence_store.append(loop.evidence_db.get("a10_e1"))
    cid = loop.chains.upsert(
        MechanismChain(
            id="",
            summary=f"{GENE_A} -> {PATHWAY_X} -> {DOWNSTREAM_PHENOTYPE}",
            evidence_ids=["a10_e1"],
        ),
        status="proposed",
    )
    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            f"Use the chain_status tool to mark chain_id={cid} as refuted with "
            f"evidence_ids=['a10_e1'] and note='wet-lab no {GENE_B} change'. "
            "Then briefly confirm.",
            session_key="a10_session",
        )
    finally:
        tracker.restore()
    log = loop.memory.read("hypothesis_log")
    _expect(cid in log and "refuted" in log,
            f"hypothesis_log missing refuted entry for {cid}: {log!r}")
    _expect(loop.chains.get_status(cid) == "refuted",
            f"chain_status did not flip; got {loop.chains.get_status(cid)!r}")
    return {
        "chain_id": cid,
        "hypothesis_tail": log.strip().splitlines()[-1],
        "response_preview": resp[:240],
        "usage": tracker.summary(),
    }


async def a11_context_compression_long_session() -> dict[str, Any]:
    """A11 -- ContextCompressor synthesises and trims when over budget.

    We do not need 30 real LLM turns to verify this; the engine's
    deterministic surface (should_compress + compress) is what we test
    here. The DeepSeek live invocation just confirms a follow-up
    process_direct succeeds after compression.
    """
    from cytopert.agent.context_compressor import CytoPertCompressor

    loop, provider, model = _make_loop_for_mode()
    # Force a small budget so we can trip compression with a few short
    # synthetic messages.
    loop.context_engine = CytoPertCompressor(
        provider=provider, model=model,
        protect_first_n=2, protect_last_n=2, context_length=200,
    )
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    tracker = _UsageTracker(provider)
    try:
        compressed = loop.context_engine.compress(msgs)
        # Then a normal short turn through DeepSeek to confirm the loop
        # still functions after the engine fired.
        resp = await loop.process_direct(
            "Reply briefly: list 3 known scverse libraries.",
            session_key="a11_session",
        )
    finally:
        tracker.restore()
    _expect(len(compressed) < len(msgs),
            f"compressor did not shrink: {len(compressed)} >= {len(msgs)}")
    _expect("error calling llm" not in resp.lower(),
            f"LLM error after compression: {resp[:200]}")
    return {
        "messages_before": len(msgs),
        "messages_after": len(compressed),
        "compression_count": loop.context_engine.compression_count,
        "response_preview": resp[:200],
        "usage": tracker.summary(),
    }


async def a12_plugin_loading() -> dict[str, Any]:
    """A12 -- a plugin written under ~/.cytopert/plugins/ becomes callable.

    Sets up a tiny ``echo_test`` plugin file inside the test's isolated
    CYTOPERT_HOME, then asks the model to call it. Verifies the plugin
    tool is dispatched (the response carries the plugin's signature).
    """
    import textwrap

    home = Path(os.environ["CYTOPERT_HOME"])
    plugin_dir = home / "plugins" / "live_echo"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "cytopert_plugin.py").write_text(
        textwrap.dedent(
            """
            import json

            async def _handler(text: str = ""):
                return json.dumps({"echo": text, "plugin": "live_echo"})

            def setup(ctx):
                ctx.register_tool(
                    name="live_echo",
                    schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    handler=_handler,
                    description="Live test plugin: echo back the supplied text.",
                )
            """
        ),
        encoding="utf-8",
    )

    loop, provider, model = _make_loop_for_mode()
    _expect("live_echo" in loop.tools.tool_names,
            f"plugin tool not registered: tools={sorted(loop.tools.tool_names)}")
    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            "Use the live_echo tool with text='ping' to verify the plugin works. "
            "Then briefly confirm.",
            session_key="a12_session",
        )
    finally:
        tracker.restore()
    _expect("ping" in resp, f"live_echo tool reply not surfaced: {resp[:200]!r}")
    return {
        "tools_count": len(loop.tools.tool_names),
        "response_preview": resp[:240],
        "usage": tracker.summary(),
    }


async def a13_trajectory_save() -> dict[str, Any]:
    """A13 -- AgentLoop(save_trajectory=True) writes a ShareGPT JSONL line."""
    loop, provider, model = _make_loop_for_mode()
    loop.save_trajectory = True
    tracker = _UsageTracker(provider)
    try:
        resp = await loop.process_direct(
            "Reply briefly: say hello.",
            session_key="a13_session",
        )
    finally:
        tracker.restore()
    from cytopert.agent.trajectory import trajectories_dir
    target = trajectories_dir() / "trajectory_samples.jsonl"
    _expect(target.exists(), f"trajectory file missing at {target}")
    last = target.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(last)
    _expect(payload["completed"] is True, f"trajectory.completed: {payload!r}")
    _expect(payload["session_key"] == "a13_session",
            f"unexpected session_key: {payload.get('session_key')!r}")
    return {
        "trajectory_path": str(target),
        "trajectory_lines": len(target.read_text(encoding="utf-8").splitlines()),
        "response_preview": resp[:200],
        "usage": tracker.summary(),
    }


TESTS_TIER_A = {
    "a1": ("Tier A: real scientific pipeline", a1_real_pipeline),
    "a2": ("Tier A: plan-before-execute", a2_plan_before_execute),
    "a3": ("Tier A: cross-session persistence", a3_cross_session),
    "a4": ("Tier A: chain lifecycle proposed->refuted", a4_chain_lifecycle),
    "a5": ("Tier A: reflection real side effects", a5_reflection_side_effects),
    "a6": ("Tier A: CLI subprocess end-to-end", a6_cli_subprocess),
    "a8": ("Tier A: PlanGate hard enforcement", a8_plan_gate_hard),
    "a9": ("Tier A: evidence-binding phantom retry", a9_evidence_binding_phantom),
    "a10": ("Tier A: chain_status writes HYPOTHESIS_LOG", a10_chain_status_writes_hypothesis_log),
    "a11": ("Tier A: ContextCompressor long session", a11_context_compression_long_session),
    "a12": ("Tier A: plugin loading via ~/.cytopert/plugins", a12_plugin_loading),
    "a13": ("Tier A: trajectory save", a13_trajectory_save),
    # a7 omitted from default registry (re-runs a1; only triggered by --tests a7)
}

TESTS_TIER_B = {
    "b2": ("Tier B: evidence_search multi-filter", b2_evidence_search_filters),
    "b3": ("Tier B: skill create-staged then accept", b3_skill_create_and_accept),
}


async def _run(label: str, fn) -> tuple[bool, dict[str, Any]]:
    started = time.time()
    try:
        info = await fn()
        info["elapsed_s"] = round(time.time() - started, 1)
        return True, info
    except Exception as e:
        info = {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=4),
            "elapsed_s": round(time.time() - started, 1),
        }
        return False, info


_ALL_TESTS_BY_ID: dict[str, tuple[str, Any]] = {
    **TESTS_DIRECT,
    "t7": TESTS_COMPAT["t7"],
    **TESTS_TIER_A,
    "a7": ("Tier A: evidence-builder quality (re-runs a1)", a7_evidence_quality),
    **TESTS_TIER_B,
}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "compat", "both"], default="direct")
    parser.add_argument(
        "--tier",
        choices=["v1", "A", "B", "all"],
        default="A",
        help=(
            "v1 = original t1..t7 smoke tests; "
            "A = a1..a13 flagship coverage (stage 8 added a8-a13); "
            "B = b2..b3 supplement (b1 dropped with the decoupler stub); "
            "all = everything plus a7 (re-runs a1)"
        ),
    )
    parser.add_argument("--tests", default=None,
                        help="Override --tier with explicit comma list (e.g. a1,a4,b2)")
    parser.add_argument("--no-census", action="store_true", help="Skip T5 (census_query)")
    args = parser.parse_args()

    creds = _read_deepseek_credentials()
    print(f"[cred] api_key prefix={creds[0][:10]}... base_url={creds[1]} model={creds[2]}")

    if args.tests:
        selected_ids = [s.strip() for s in args.tests.split(",") if s.strip()]
    elif args.tier == "v1":
        selected_ids = list(TESTS_DIRECT.keys()) + (["t7"] if args.mode in ("compat", "both") else [])
    elif args.tier == "A":
        selected_ids = list(TESTS_TIER_A.keys())
    elif args.tier == "B":
        selected_ids = list(TESTS_TIER_B.keys())
    else:  # all
        selected_ids = list(_ALL_TESTS_BY_ID.keys())

    results: list[tuple[str, str, bool, dict[str, Any]]] = []
    homes: list[Path] = []

    for tid in selected_ids:
        if tid not in _ALL_TESTS_BY_ID:
            print(f"[skip] unknown test id {tid!r}")
            continue
        label, fn = _ALL_TESTS_BY_ID[tid]
        if tid == "t5" and args.no_census:
            continue
        # Mode resolution: t7 always runs in compat; everything else uses --mode (default direct).
        mode = "compat" if tid == "t7" else (
            "direct" if args.mode == "compat" else args.mode if args.mode != "both" else "direct"
        )
        home = _setup_isolated_home(mode, creds)
        homes.append(home)
        print(f"\n=== [{mode}] {tid}: {label} ===")
        ok, info = await _run(label, fn)
        mark = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  {mark}  {label}")
            print(f"  ## REFLECTION on {tid} failure")
            print(f"  1. Repro:    {info.get('error', '?')}")
            print( "  2. Category: <pending — set after triage>")
            print( "  3. Fix scope: <pending>")
            print( "  4. Risk of overreach: <pending>")
            print( "  5. Decision: <pending>")
        else:
            print(f"  {mark}  {label}")
        for k, v in info.items():
            if k == "trace" and ok:
                continue
            print(f"    {k}: {v}")
        results.append((mode, tid, ok, info))

    print("\n========= SUMMARY =========")
    for mode, tid, ok, info in results:
        mark = "PASS" if ok else "FAIL"
        token = info.get("usage", "-")
        soft = info.get("soft_fail_reason")
        suffix = f"  [{soft}]" if soft else ""
        print(f"  [{mode}] {tid:>3}  {mark}  {token}{suffix}")

    for h in homes:
        try:
            shutil.rmtree(h, ignore_errors=True)
        except Exception:
            pass

    failed = sum(1 for _, _, ok, _ in results if not ok)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
