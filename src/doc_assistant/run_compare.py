"""
Side-by-side evaluation runner for LangGraph vs LlamaIndex Pokemon assistants.

Suite 1: Golden queries -- correctness (keyword check) + wall-clock latency + LLM call count
Suite 2: Multi-turn chain -- designed to expose each framework's gaps

Run:  python -m src.doc_assistant.run_compare
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

GOLDEN_QUERIES = [
    "What types are super effective against Water?",
    "Tell me about Thunderbolt's stats and what it's good against.",
    "What's a good strategy for a Fire type facing a Water/Ground Pokemon?",
    "Summarize the type chart for Dragon types.",
]

GOLDEN_KEYWORDS: dict[int, list[str]] = {
    0: ["Electric", "Grass"],
    1: ["Electric", "90", "Special"],
    2: ["Ground"],
    3: ["Dragon"],
}

MULTI_TURN_CHAIN = [
    ("session-mt", "What types are super effective against Water?"),
    ("session-mt", "What about its resistances?"),
    ("session-mt", "Which of those types also resist Dragon?"),
]

MULTI_TURN_KEYWORDS: dict[int, list[str]] = {
    0: ["Electric", "Grass"],
    1: ["Water"],       # Q2 should discuss Water's resistances
    2: ["Dragon"],      # Q3 should cross-reference
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query: str
    answer: str
    latency_ms: float
    keywords_expected: list[str] = field(default_factory=list)
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    passed: bool = False


def check_keywords(answer: str, keywords: list[str]) -> tuple[list[str], list[str], bool]:
    answer_lower = answer.lower()
    found = [k for k in keywords if k.lower() in answer_lower]
    missing = [k for k in keywords if k.lower() not in answer_lower]
    return found, missing, len(missing) == 0


# ---------------------------------------------------------------------------
# Suite runners
# ---------------------------------------------------------------------------

async def run_suite1_for_stack(ask_fn, stack_name: str) -> list[QueryResult]:
    """Run golden queries against one stack."""
    results = []
    for i, query in enumerate(GOLDEN_QUERIES):
        t0 = time.perf_counter()
        answer = await ask_fn(query, session_id=f"suite1-{stack_name}-{i}")
        latency = (time.perf_counter() - t0) * 1000

        keywords = GOLDEN_KEYWORDS.get(i, [])
        found, missing, passed = check_keywords(answer, keywords)

        results.append(QueryResult(
            query=query,
            answer=answer,
            latency_ms=latency,
            keywords_expected=keywords,
            keywords_found=found,
            keywords_missing=missing,
            passed=passed,
        ))
    return results


async def run_suite2_for_stack(ask_fn, stack_name: str) -> list[QueryResult]:
    """Run multi-turn chain against one stack (single session)."""
    results = []
    for i, (session_id, query) in enumerate(MULTI_TURN_CHAIN):
        sid = f"{session_id}-{stack_name}"
        t0 = time.perf_counter()
        answer = await ask_fn(query, session_id=sid)
        latency = (time.perf_counter() - t0) * 1000

        keywords = MULTI_TURN_KEYWORDS.get(i, [])
        found, missing, passed = check_keywords(answer, keywords)

        results.append(QueryResult(
            query=query,
            answer=answer,
            latency_ms=latency,
            keywords_expected=keywords,
            keywords_found=found,
            keywords_missing=missing,
            passed=passed,
        ))
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_header(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_comparison(label: str, lg_results: list[QueryResult], li_results: list[QueryResult]):
    print_header(label)

    for i, (lg, li) in enumerate(zip(lg_results, li_results)):
        print(f"\n  Q{i+1}: {lg.query}")
        print(f"  {'─'*74}")

        print(f"  {'':30s} {'LangGraph':20s} {'LlamaIndex':20s}")
        print(f"  {'─'*74}")

        lg_status = "PASS" if lg.passed else "FAIL"
        li_status = "PASS" if li.passed else "FAIL"
        print(f"  {'Correctness':30s} {lg_status:20s} {li_status:20s}")

        print(f"  {'Latency':30s} {lg.latency_ms:>8.0f} ms{'':10s} {li.latency_ms:>8.0f} ms")

        if lg.keywords_missing:
            print(f"  {'LG missing keywords':30s} {', '.join(lg.keywords_missing)}")
        if li.keywords_missing:
            print(f"  {'LI missing keywords':30s} {', '.join(li.keywords_missing)}")

        print(f"\n  LangGraph answer (first 300 chars):")
        print(f"    {lg.answer[:300]}")
        print(f"\n  LlamaIndex answer (first 300 chars):")
        print(f"    {li.answer[:300]}")
        print()


def print_suite2_detail(lg_results: list[QueryResult], li_results: list[QueryResult]):
    """Extra detail for suite 2 showing the multi-turn gap."""
    print_header("Suite 2: Multi-Turn Analysis")

    print("\n  This suite exposes each framework's gaps:")
    print("  - LangGraph retrieves against raw follow-up strings (no condensing)")
    print("  - LlamaIndex condenses automatically but internals are opaque")
    print()

    for i, (lg, li) in enumerate(zip(lg_results, li_results)):
        lg_status = "PASS" if lg.passed else "FAIL"
        li_status = "PASS" if li.passed else "FAIL"
        marker = ""
        if i > 0:
            marker = "  <-- follow-up (pronoun reference)"
        print(f"  Q{i+1} [{lg_status}/{li_status}] {lg.query}{marker}")

    print(f"\n  Summary:")
    lg_pass = sum(1 for r in lg_results if r.passed)
    li_pass = sum(1 for r in li_results if r.passed)
    print(f"    LangGraph:  {lg_pass}/{len(lg_results)} passed")
    print(f"    LlamaIndex: {li_pass}/{len(li_results)} passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    from src.doc_assistant import langgraph_assistant, llamaindex_assistant

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env or environment")
        return

    # Warm up the shared index once
    print("Building/loading shared index...")
    from src.doc_assistant.shared.indexer import get_index
    get_index()
    print("Index ready.\n")

    # Suite 1
    print("Running Suite 1: Golden Queries...")
    lg_s1 = await run_suite1_for_stack(langgraph_assistant.ask, "langgraph")
    li_s1 = await run_suite1_for_stack(llamaindex_assistant.ask, "llamaindex")
    print_comparison("Suite 1: Golden Queries", lg_s1, li_s1)

    # Suite 2
    print("\nRunning Suite 2: Multi-Turn Chain...")
    lg_s2 = await run_suite2_for_stack(langgraph_assistant.ask, "langgraph")
    li_s2 = await run_suite2_for_stack(llamaindex_assistant.ask, "llamaindex")
    print_comparison("Suite 2: Multi-Turn Chain", lg_s2, li_s2)
    print_suite2_detail(lg_s2, li_s2)

    # Extension tests
    print_header("Extension 1: Meta-question (control-flow)")
    meta_q = "What can you help me with?"
    lg_meta = await langgraph_assistant.ask(meta_q, session_id="ext1-lg")
    li_meta = await llamaindex_assistant.ask(meta_q, session_id="ext1-li")
    print(f"\n  Query: {meta_q}")
    print(f"\n  LangGraph (should skip retrieval, use direct_respond node):")
    print(f"    {lg_meta[:300]}")
    print(f"\n  LlamaIndex (should skip retrieval, pre-processing check):")
    print(f"    {li_meta[:300]}")

    print_header("Extension 2: Ability routing (data-centric)")
    ability_q = "What does the Intimidate ability do and which Pokemon have it?"
    lg_ability = await langgraph_assistant.ask(ability_q, session_id="ext2-lg")
    li_ability = await llamaindex_assistant.ask_with_routing(ability_q)
    ability_kw = ["Intimidate", "Attack"]
    lg_found, lg_miss, lg_pass = check_keywords(lg_ability, ability_kw)
    li_found, li_miss, li_pass = check_keywords(li_ability, ability_kw)
    print(f"\n  Query: {ability_q}")
    print(f"\n  {'':30s} {'LangGraph':20s} {'LlamaIndex':20s}")
    print(f"  {'─'*74}")
    print(f"  {'Correctness':30s} {'PASS' if lg_pass else 'FAIL':20s} {'PASS' if li_pass else 'FAIL':20s}")
    if lg_miss:
        print(f"  {'LG missing keywords':30s} {', '.join(lg_miss)}")
    if li_miss:
        print(f"  {'LI missing keywords':30s} {', '.join(li_miss)}")
    print(f"\n  LangGraph answer (first 300 chars):")
    print(f"    {lg_ability[:300]}")
    print(f"\n  LlamaIndex answer (first 300 chars):")
    print(f"    {li_ability[:300]}")

    # Code volume comparison
    print_header("Code Volume Comparison (with extensions)")
    import pathlib
    base = pathlib.Path("src/doc_assistant")
    for name in ["langgraph_assistant.py", "llamaindex_assistant.py"]:
        p = base / name
        if p.exists():
            lines = p.read_text().splitlines()
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#") and not l.strip().startswith('"""') and not l.strip().startswith("'''")]
            print(f"  {name:35s} total={len(lines):3d} lines   code={len(code_lines):3d} lines")


if __name__ == "__main__":
    asyncio.run(main())
