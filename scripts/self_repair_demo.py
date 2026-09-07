"""Explicit fault injection -> actual real LLM repair -> validated PASS."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_acceptance import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["auto", "openai", "local"], default="auto")
    parser.add_argument("--output", default="examples/real_self_repair")
    args = parser.parse_args()
    run(Path(args.output), "first", args.provider)
    run(Path(args.output), "repair", args.provider)

