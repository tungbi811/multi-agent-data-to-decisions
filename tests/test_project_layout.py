import importlib
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multi_agents_is_a_valid_package():
    assert not (ROOT / "multi-agents").exists()
    assert (ROOT / "multi_agents" / "__init__.py").is_file()
    assert importlib.import_module("multi_agents") is not None


def test_project_uses_uv_as_its_only_dependency_source():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["requires-python"] == ">=3.12,<3.14"
    assert metadata["tool"]["uv"]["package"] is False
    assert (ROOT / "uv.lock").is_file()
    assert not (ROOT / "poetry.lock").exists()
    assert not (ROOT / "requirements.txt").exists()
