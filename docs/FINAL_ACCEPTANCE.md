# Final Acceptance Snapshot

Final evidence is generated against the current Git commit by `scripts/annotate_evidence.py` and checked by `scripts/verify_evidence.py`. Raw run reports and large benchmark payloads are not treated as final claims unless they are re-generated and annotated at the final commit.

The workflow distinguishes:

- `free_form_llm`: LLM emits Python, then strict protocol/semantic/sandbox gates;
- `structured_synthesis`: LLM emits `AlgorithmImplementationSpec` CodeIR and a deterministic compiler emits Python;
- `controlled_real_llm_interface_repair`: deliberately broken interface fixture, not natural autonomous generation;
- natural CodeGen evidence: remains failed unless final sandbox passes;
- `template_fallback`: safety recovery, never reported as LLM success.

Final checks:

```bash
python -m pytest -q
python scripts/verify_evidence.py
git status --short
```

