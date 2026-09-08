# Technical Audit — 2026-09-07 re-audit

## Scope and actual baseline
Baseline checkout: main, HEAD 7dea7e6.
Read application modules, tests, demo/benchmark/evidence scripts and dependency configuration.
Baseline included five modified tracked files and two additional modules; the audit preserved their original contents.

Python 3.10 (ada_qwen), LLM_PROVIDER=mock, ENABLE_LOCAL_EMBEDDING=0:
- pytest -q: **6 collection errors**, exit 2, 3.48 seconds.
- scripts/run_demo.py: **failed**, exit 1.
- Both fail at generator_agent.py:43: misindented gate causes SyntaxError.
Earlier documentation's 12 passing tests refers to a different checkout.

## Verified pre-remediation findings

| Area | Actually implemented | Gap and consequence |
|---|---|---|
| Requirement | LLM JSON, Pydantic validation/retry, regex/CSV fallback | Prompt omits full schema; LLM profile can replace measured dataset facts; regex candidates survive successful understanding; PR-AUC conflated with ROC-AUC. |
| LLM | OpenAI-compatible API, local Transformers router, mock | 12-second API timeout; credential-bearing error strings unredacted; auto can silently become mock; only two code candidates use LLM; template recovery can make demo PASS. |
| Configuration | Environment assignment reader | Inline base_url/api_key syntax parsed incorrectly; configuration parsing and redaction required verification. |
| Extraction | Markdown/TXT, AST Python and JSON reports -> LLM entities/relations -> graph | Separate ingest command, absent normal bootstrap; evidence spans unchecked; duplicate entities across chunks lose relation endpoint aliases; report config/runtime associations incomplete; requires whole-document metrics in every chunk. |
| Graph | Real NetworkX bounded traversal and serializer | Anchors use arbitrary full JSON text across all node types; Chinese tokens dropped; hub traversal crosses incompatible tasks; graph converted per anchor; no relation-path trace; repairs two edges away absent algorithm summaries. |
| Hybrid | TF-IDF or optional local embedding retrieval | No actual graph/document merge scoring; re-embeds unchanged corpus; database query for every graph node. |
| Experience | Task/feature/rows/balance similarity, recency, success/exploration priors | Only winner stored as ValidationRun; successful alternatives vanish; failed variants fragment algorithm statistics; current profile may omit measured facts; resource similarity weak. |
| Planner | LLM advice sees graph/history; deterministic scorer + execution | Only candidate names affect execution; LLM reasons/preprocessing/config decisions discarded. Advice can eliminate exploration. |
| Search | Already expands 12 algorithm/preprocessing/config states | Fixed preset one-level search, not MCTS; workflow caps beam by base algorithm count; single-algorithm task runs one config; unknown plugin silently uses wrong estimator template. |
| Code | Real free-form generation with static gate | Prompt lacks retrieved evidence and constraints; duplicate weaker safety gate; metadata/predict_proba unchecked; current file SyntaxError. |
| Repair | LLM Critic and code rewrite branches really run | Critic unvalidated; no structured repair strategy; no immutable per-round versions/results; no-op fallback claims regeneration; successful repair loses original failure. |
| Validation | Protocol, subprocess, split, output bounds, three trains, timeout | Trusts generated evaluate metrics, permitting fabricated scores; cross-seed variance only displayed; memory/latency constraints not enforced; unbound df on missing CSV; mean() fails string labels. |
| Sandbox | python -I, temporary CWD, AST, timeout | Inherits secrets; pandas/numpy I/O and reflection bypass AST; resource helper missing resource import and disabled by default; no network/filesystem restriction. Prototype only. |
| Curator/version | SQLite writes, GraphML export, final version node | Only winner validation, RELATED_TO instead of VALIDATES algorithm; missing dataset/capability/environment materialization, SATISFIES and parent versions. |
| Multi-Agent | Specialist modules invoked by synchronous coordinator | No shared typed messages/tool capability enforcement; candidate loop in workflow; inconsistent telemetry. It is cooperating specialist workflow, not distributed autonomous agents. |
| UI | CLI/FastAPI and HTML | Hardcoded mock, raw JSON dump; no relevant subgraph/run/failure visualization or generated-code inspection. |
| Evidence | Historical scripts/artifacts exist | The controlled prior injection demo injects 0.99 rather than measuring a workflow run (current filename: controlled_prior_injection_demo.py); annotate_evidence relabels old commits without rerunning; verifier lacks artifact hashes and execution lineage checks. |
| Plugins | Registry feeds planning/search | Unknown algorithm falls through to GradientBoosting; registration alone does not prove executable extension. |

## Core workflow assessment
LLM participates in requirement, advisor, top-budget code proposals, failure diagnosis and repair, but fallbacks can produce the final PASS. KG does participate through traversal, although entity linking and path filtering are weak. Written winner ValidationRuns affect subsequent deterministic priors; existing automated tests mostly insert hand-authored runs, not Task A -> Task B. Multi-Agent responsibilities are partly real; message contracts and evidence transfer are incomplete. Beam expands real presets and executes selected states, but its breadth and diversity policy are restricted. Existing sandbox/stability are useful prototype checks, not hostile-code isolation or statistically robust generalization evidence.

## Gaps against scoring criteria
- Technical (40%): restore execution, require truthful real-provider acceptance, independent metrics, complete candidate/repair provenance, actual learned-history planning loop.
- Code quality (30%): central configuration/telemetry/sanitization, explicit contracts, split candidate execution from orchestration, isolated regression tests, reproducible dependencies.
- Innovation (15%): relation-aware GraphRAG with path evidence, contextual priors plus exploration, LLM decisions affecting executed configurations, reusable failure lessons. Controlled fixtures, experience reuse and finite beam search are evaluated separately.
- Completeness (15%): measured real LLM, self-repair and cross-domain runs; hash-bound artifacts; usable UI and accurate documentation.

## Improvement scope and acceptance criteria
1. Versioned audit, baseline source and execution logs.
2. Secure API parsing, truthful provider failures, structured agents and full traces.
3. Provenance-grounded extraction -> relation-aware graph -> serialized Planner/Coder evidence.
4. Persist all candidates/rounds, context-sensitive priors, exploration, immutable code versions.
5. Independently compute metrics, enforce constraints, harden prototype sandbox.
6. Run full tests; real LLM tasks A/B, fail->LLM repair->PASS and second task type.
7. Artifact hashes bind evidence to actual source digest; never retroactively relabel older experiments.
8. Keep actual model execution, controlled fixtures and mock tests explicitly distinguishable.

This audit describes the baseline version. The implementation results and measured acceptance are documented in [FINAL_ACCEPTANCE.md](FINAL_ACCEPTANCE.md).
