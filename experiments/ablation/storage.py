"""Read-only source export, isolated stores, protocol commitments and frozen-file guard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

TABLES = ("capabilities", "algorithms", "validation_runs", "experiences", "knowledge_items", "graph_edges")
MEMORY_TYPES = {"ValidationRun", "ValidationResult", "FailureExperience", "RepairExperience",
                "AlgorithmVersion", "FailureCase", "OptimizationExperience"}
MEMORY_FIELDS = {"historical_metrics", "historical_runs", "validation_history",
                 "failure_experiences", "repair_experiences", "historical_statistics"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def git_commit(root: Path) -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def export_snapshot(source: Path, output: Path):
    """Source is opened read-only and never passed to KnowledgeStore."""
    with sqlite3.connect(f'file:{source.resolve()}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        snapshot = {table: [dict(row) for row in conn.execute(f'SELECT * FROM {table}')]
                    for table in TABLES}
    atomic_json(output, snapshot)
    return snapshot


def _contains_memory(value) -> bool:
    if isinstance(value, dict):
        return value.get('type') in MEMORY_TYPES or value.get('node_type') in MEMORY_TYPES or any(
            _contains_memory(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_memory(child) for child in value)
    return False


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(child) for key, child in value.items() if key not in MEMORY_FIELDS}
    if isinstance(value, list):
        return [_clean(child) for child in value]
    return value


def without_experience(snapshot: dict) -> tuple[dict, list[str]]:
    """Remove learned/cold numeric priors from every retrieval channel, including sources.

    Domain descriptions and expert preprocessing rules remain. Experiment-report
    sources and their extracted entities are removed as a unit (not rewritten).
    """
    output = json.loads(json.dumps(snapshot))
    memory_ids = set()
    run_ids = {row['id'] for row in snapshot['validation_runs']}
    for row in snapshot['validation_runs']:
        payload = json.loads(row['payload'])
        memory_ids.add(row['id'])
        memory_ids.update(payload[key] for key in ('version_id', 'dataset_id', 'config_id', 'capability_id')
                          if payload.get(key))
    memory_ids.update(row['id'] for row in snapshot['experiences'])
    for row in snapshot['knowledge_items']:
        payload = json.loads(row['payload'])
        if row['node_type'] in MEMORY_TYPES or _contains_memory(payload):
            memory_ids.add(row['id'])
    # Remove entities supported exclusively by removed experiment sources.
    for row in snapshot['graph_edges']:
        if row['source'] in memory_ids and row['relation'] == 'SUPPORTS':
            memory_ids.add(row['target'])
    output['validation_runs'], output['experiences'] = [], []
    for table in ('capabilities', 'algorithms', 'knowledge_items'):
        rows = []
        for row in output[table]:
            payload = json.loads(row['payload'])
            if (row['id'] in memory_ids or payload.get('origin') == 'measured_workflow'
                or any(run_id in row['payload'] for run_id in run_ids)):
                memory_ids.add(row['id'])
                continue
            row['payload'] = json.dumps(_clean(payload), ensure_ascii=False)
            rows.append(row)
        output[table] = rows
    retained = {row['id'] for table in TABLES[:-1] for row in output[table]}
    output['graph_edges'] = [row for row in output['graph_edges']
                             if row['source'] in retained and row['target'] in retained]
    return output, sorted(memory_ids)


def import_snapshot(snapshot: dict, destination: Path):
    from app.knowledge.store import KnowledgeStore
    if destination.exists():
        raise FileExistsError('trial database must be new')
    store = KnowledgeStore(destination)
    with store._connect() as conn:
        for table in TABLES:
            for row in snapshot[table]:
                columns = list(row)
                # The only SQL identifiers come from a fixed exported table schema.
                valid = {r[1] for r in conn.execute(f'PRAGMA table_info({table})')}
                if not set(columns) <= valid:
                    raise ValueError('unexpected snapshot columns')
                conn.execute(f'INSERT INTO {table} ({",".join(columns)}) VALUES ({",".join("?" for _ in columns)})',
                             tuple(row[column] for column in columns))
    store.refresh()
    store.export_graph()
    return store


def protect_files(root: Path) -> dict[str, str]:
    prefixes = ('app/', 'examples/acceptance_real_20260907/',
                'examples/acceptance_text_20260908/', 'data/uci_sentiment/')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root, text=True).split('\0')
    paths = {name for name in tracked if name.startswith(prefixes)}
    # Also guard the two untracked original acceptance databases.
    for name in ('examples/acceptance_real_20260907/workspace/knowledge.sqlite',
                 'examples/acceptance_text_20260908/workspace/knowledge.sqlite'):
        if (root / name).exists():
            paths.add(name)
    return {name: sha(root / name) for name in sorted(paths)}


def verify_guard(root: Path, guard: dict[str, str]):
    differences = [name for name, digest in guard.items()
                   if not (root / name).is_file() or sha(root / name) != digest]
    if differences:
        raise RuntimeError(f'frozen/core file guard failed: {differences}')


def assert_experiment_output(root: Path, output: Path):
    allowed = (root / 'experiments/ablation/results').resolve()
    if not output.resolve().is_relative_to(allowed) or output.resolve() == allowed:
        raise ValueError('output must be a dedicated directory under experiments/ablation/results')
