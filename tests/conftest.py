"""Tests must never load a real model or mutate the user's working knowledge graph."""
from pathlib import Path
import shutil
import pytest


@pytest.fixture(autouse=True)
def isolate_default_workspace(tmp_path, monkeypatch):
    root = tmp_path / "default_workspace"
    (root / "data").mkdir(parents=True)
    source = Path(__file__).parents[1] / "data/business_material.md"
    shutil.copy(source, root / "data/business_material.md")
    monkeypatch.setenv("AI_FACTORY_WORKSPACE", str(root))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("ENABLE_LOCAL_EMBEDDING", "0")
