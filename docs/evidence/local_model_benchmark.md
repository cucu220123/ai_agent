# Local Model Benchmark

Strict success means schema/runtime gates passed; an LLM call alone is not success.

| Model | Weight GB | Requirement | Planner | Executable code | Repair | Overall | Latency s | Peak GPU MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 3.09 | FAIL | FAIL | FAIL | FAIL | 0.00 | 67.6 | 3041 |
| Qwen2.5-Coder-3B-Instruct | 6.17 | PASS | FAIL | FAIL | PASS | 0.50 | 128.6 | 6004 |
| Qwen2.5-7B-Instruct | 15.23 | PASS | FAIL | FAIL | FAIL | 0.25 | 92.0 | 14722 |
| Qwen3-8B | 16.38 | PASS | PASS | FAIL | PASS | 0.75 | 392.3 | 16084 |
| Qwen2.5-14B-Instruct | 29.54 | PASS | PASS | FAIL | PASS | 0.75 | 211.4 | 28538 |
| Qwen3-Coder-30B-A3B-Instruct | 61.07 | FAIL | FAIL | FAIL | FAIL | 0.00 | 362.9 | 50638 |

Selected routing: Qwen2.5-14B-Instruct for structured language tasks; Qwen2.5-Coder-3B-Instruct for code/repair. Qwen3-8B and Qwen2.5-14B both reached 0.75 strict benchmark success but are slower; Qwen3-Coder-30B reached 0.75 on a GPU4 run but required about 745 seconds and remains optional for exclusive-GPU deployment. The earlier GPU0 shared run failed with CUDA OOM and is retained as a resource-limit evidence case.
