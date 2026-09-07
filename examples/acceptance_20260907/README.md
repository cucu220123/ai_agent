# Development attempts — not final acceptance

This directory retains genuine failed or interrupted real-model attempts during the audit. They are not counted as passing acceptance and must not be relabeled as current successful runs.

- `development_logs/baseline-*`: initial test/demo failures before remediation.
- Other development logs: actual JSON, protocol, import, preprocessing and repair failures.
- `pre_30b_workspace`: archived incomplete runs using Qwen2.5-Coder-3B-Instruct; some generation calls were interrupted while improving the host contract. Original absolute paths in these archived records describe their execution location at that time.
- Earlier workspace artifacts: instruction-model code generation trials, including rejected dependencies and runtime errors.

The verified fresh experiment is in `../acceptance_real_20260907/`. Only its manifest, complete reports, immutable source versions and read-only verifier establish acceptance.
