from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_the_implemented_workflow_without_old_claims():
    readme = (ROOT / "README.md").read_text()
    assert "uv sync --locked" in readme
    assert "docker build -f docker/executor.Dockerfile" in readme
    assert "gpt-4.1-mini" in readme
    assert "AUTO_DS_MODEL" in readme
    assert "structured/tabular CSV" in readme
    assert "artifacts/runs/<run-id>" in readme
    assert "pip install -r requirements.txt" not in readme
    assert "Python 3.11" not in readme
    assert "GPT-4o-mini" not in readme
    assert "ensures code runs safely" not in readme


def test_evaluation_template_requires_honest_run_evidence():
    template = (ROOT / "docs" / "evaluation-template.md").read_text()
    for field in (
        "Dataset",
        "Prompt",
        "Agent trace",
        "Generated code",
        "Recommendation",
        "Runtime",
        "Failures",
    ):
        assert field in template
