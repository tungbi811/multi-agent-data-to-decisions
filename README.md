# Auto DS Agents

Auto DS Agents is a local capstone prototype for structured/tabular CSV data.
Four specialised AG2 agents collaborate to turn a business requirement and one
or more uploaded datasets into analysis, executable Python, and a
stakeholder-facing recommendation.

The agents are:

- **Business Analyst** — interprets the business requirement and structures the
  objective.
- **Data Scientist** — plans the analysis and selects suitable techniques.
- **Coder** — generates complete Python scripts and executes them in Docker.
- **Business Translator** — converts analytical results into an actionable,
  non-technical recommendation.

The default agent model is `gpt-4.1-mini`. Set `AUTO_DS_MODEL` before launching
the application to override that default.

## Scope

The prototype accepts structured/tabular CSV input through its Streamlit
interface. It is not a production AutoML platform and does not support
unstructured data, real-time streaming, enterprise deployment, or proprietary
system integrations.

## Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker with a running local daemon
- An OpenAI API key for a manual analysis run

## Setup

Install the locked application and development dependencies:

```bash
uv sync --locked --python 3.12
```

Build the locked, network-isolated code-executor image:

```bash
docker build -f docker/executor.Dockerfile -t auto-ds-executor:0.1 .
```

Launch the local interface:

```bash
uv run streamlit run main.py
```

Enter the OpenAI API key only in the local Streamlit password field. Do not add
it to source, logs, artifacts, screenshots, commits, Docker images, uploaded
data, or generated-code containers.

## Verification

Run the offline test suite:

```bash
uv run pytest -m "not docker"
```

Run the opt-in Docker isolation tests after building the executor image:

```bash
RUN_DOCKER_TESTS=1 uv run pytest tests/test_docker_smoke.py
```

Automated tests do not make live OpenAI calls.

## Runtime safety and evidence

Generated Python runs in a restricted Docker container with no outbound
network, no API key, a read-only container root filesystem, and access only to
the current run workspace. Docker reduces exposure to the host, but it does not
make this prototype production-ready. Treat uploads, generated code, and
results as untrusted. Use only public, non-sensitive sample data, and stop if a
generated script attempts an unexpected file, network, or destructive
operation.

At completion, generated code, outputs, trace, runtime, and failure evidence
are retained under `artifacts/runs/<run-id>`. Uploaded CSV copies and the
temporary run workspace are removed. Use
[`docs/evaluation-template.md`](docs/evaluation-template.md) to record a manual
rerun without turning proposed targets into claimed results.

## Illustrative example

A house-price request might ask how to estimate market value from property
features. An illustrative recommendation could be to prioritise property
quality improvements in selected neighbourhoods. This is an example of the
expected output style, not a measured benchmark or a recorded project result.

## Contributors

- Monika Shakya
- Van Thang Doan
- Yamuna G C
- Linh Chi Tong
- Szu-Yu Lin
- Duy Tung Nguyen — System Developer

**License:** MIT
