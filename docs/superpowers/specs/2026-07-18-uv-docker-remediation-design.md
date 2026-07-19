# UV Migration and Project Remediation Design

## Objective

Make Auto DS Agents reproducible, testable, and safe enough for a local
portfolio demonstration while preserving its scope as a UTS tabular-data
capstone prototype. Complete the current package rename in place, replace
Poetry with uv, isolate generated-code execution in Docker, fix the defects
identified in the repository review, and capture honest run evidence.

## Scope and constraints

- Complete the current `multi-agents/` to `agents/` rename without
  reverting or modifying unrelated dataset renames or the existing log diff.
- Support Python `>=3.12,<3.14`.
- Use `pyproject.toml` and `uv.lock` as the only dependency sources.
- Keep AG2 pinned to `0.9.9` unless verification proves a narrowly scoped
  compatibility change is necessary.
- Support structured/tabular CSV data only.
- Do not make live OpenAI calls in automated tests.
- Never store an OpenAI API key in source, logs, artifacts, screenshots,
  commits, Docker images, or generated-code containers.
- Do not claim proposed performance or reliability targets as achieved.
- Preserve the documented team attribution and Duy Tung Nguyen's System
  Developer role.

## Architecture

### Source layout

The agent package will be named `agents` and contain a conventional
`__init__.py`. Imports in `main.py`, tests, and agent modules will consistently
use that package name. The package export list will contain strings rather than
class objects.

Configuration shared by agents will live in one Python module. Agent models
will default to `gpt-4.1-mini`, with `AUTO_DS_MODEL` as the optional
environment-variable override. The unused JSON LLM configuration and the
separate markdown-formatting LLM client will be removed. Markdown conversion
will be deterministic and local.

### Dependency management

`pyproject.toml` will use PEP 621 metadata, including `requires-python`, and uv
dependency groups for test and lint tooling. The application is not intended
to be published as a Python distribution, so uv will manage it as an
application rather than relying on Poetry package discovery.

The supported commands will be:

```bash
uv sync --locked
uv run pytest
uv run streamlit run main.py
```

`poetry.lock`, Poetry-specific configuration, and `requirements.txt` will be
removed after `uv.lock` is generated and verified.

### Analysis runtime

Each analysis run will receive a unique identifier and a dedicated workspace.
Uploaded CSVs, generated code, outputs, and evidence will be separated within
that workspace. The Streamlit session will own the active runtime and clean it
up on completion, restart, or failure.

The Coder will use AG2's Docker command-line executor rather than the current
host-local Jupyter server. A project image named `auto-ds-executor:0.1` will
contain the pinned Python data-science dependencies. Each code execution will
have a 300-second timeout. Its container will:

- receive only the current run workspace as a bind mount;
- receive no OpenAI API key or other host credentials;
- run without outbound network access;
- be limited to 2 GiB of memory, two CPUs, and 256 processes;
- be stopped and removed when the run ends.

Command-line Docker execution does not retain in-memory notebook variables.
Coder prompts will therefore require each generated step to be a complete,
reproducible Python script that reads required inputs from the run workspace.
Files written inside the workspace may be used by later steps.

### Evidence retention

Before Docker executes a generated script, the application will record it in
the temporary run workspace as `code/step-NNN.py`. The run will also collect
outputs, an agent trace, runtime, and structured failure records. On
completion, those retained files will be copied to
`artifacts/runs/<run-id>/`, while uploaded CSV copies and the remaining
temporary workspace are removed.

No artifact may contain the OpenAI API key. Dataset evidence will use the
original filename plus a content hash and basic non-sensitive shape metadata,
not a retained copy of the uploaded data.

## Data flow

1. Streamlit calls `st.set_page_config` before any other Streamlit command and
   initializes session state.
2. The sidebar collects a locally entered OpenAI API key, CSV uploads, and the
   business requirement.
3. Upload handling rejects files larger than 50 MiB and unsafe filenames, then
   writes each CSV to the active run's dataset directory outside the source
   tree.
4. `GroupChat` receives an explicit allowlist of uploaded paths.
5. `get_data_info` resolves requested paths and rejects anything outside the
   allowlist before reading CSV metadata.
6. Business Analyst, Business Translator, Data Scientist, and Coder exchange
   messages using corrected tool and target names.
7. The Coder submits a complete script. The application records the script and
   then executes it automatically in the restricted Docker container.
8. Results return to the Data Scientist and Business Translator. Events are
   displayed in Streamlit and appended to the run trace.
9. Completion or restart stops the container, removes uploaded CSV copies, and
   finalizes the evidence record.

## Error handling

- Unsafe filenames, oversized files, invalid CSVs, and paths outside the run
  allowlist fail before agent execution with a user-facing message.
- Docker-unavailable and executor-image-missing failures stop the run with a
  setup-oriented message. There is no automatic host-local fallback.
- Code is recorded before execution, so failed attempts remain auditable.
- An execution exception restarts the executor once and retries the same
  recorded script. A second exception returns a controlled failure rather than
  escaping from the tool handler.
- Nonzero command results include the useful error output rather than only its
  first two lines.
- Streamlit handles exhausted event iterators, malformed tool arguments,
  unknown event types, and cleanup failures without losing the retained trace.
- The group chat is capped at 60 rounds as a cost and termination guard.

## Testing strategy

Automated tests will not require an OpenAI API key or make network calls.

Focused tests will cover:

- `agents` package import and string-based `__all__` exports;
- PEP 621/uv project metadata and absence of conflicting dependency sources;
- safe upload filenames, size limits, unique workspaces, and cleanup;
- dataset allowlisting and rejection of out-of-scope paths;
- sequential code recording and evidence finalization;
- successful execution, nonzero results, one retry, and double failure;
- corrected tool names, agent targets, and central model configuration;
- Streamlit startup without creating an executor or requiring an API key;
- event exhaustion and malformed tool-call handling.

A separate Docker smoke test will build the executor image, execute a small
Python script, and verify that:

- the host project is not visible beyond the mounted run workspace;
- `OPENAI_API_KEY` is absent inside the container;
- outbound network access is unavailable;
- generated code and output remain after the container is removed.

The live multi-agent workflow remains a manual verification because its API
key must be entered locally in the UI. The manual check will use a
non-sensitive sample dataset and record the dataset hash, prompt, trace,
generated code, recommendation, runtime, and failures.

## Documentation

The README will:

- use uv setup and run instructions;
- state the active model configuration accurately;
- describe Docker-isolated generated-code execution without claiming broader
  production security or enterprise readiness;
- document tabular CSV scope and local-demo limitations;
- distinguish illustrative examples and proposed evaluation targets from
  measured results;
- document saved run evidence and the manual verification process;
- preserve the complete team attribution.

An evaluation template will describe the required evidence for a defensible
rerun without fabricating results.

## Completion criteria

- `uv sync --locked` succeeds with Python 3.12.
- All automated tests pass without an API key.
- Static compilation/import checks pass.
- The Docker smoke test passes on a machine with Docker available.
- The current incomplete package rename is resolved without disturbing
  unrelated dirty worktree changes.
- No Poetry or requirements dependency source remains.
- Documentation matches implemented behavior and evidence limits.
- No live-result claim is made unless the user completes the documented manual
  run locally.
