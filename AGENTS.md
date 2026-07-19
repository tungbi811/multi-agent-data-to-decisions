# Auto DS Agents: Working Context

## Purpose

Auto DS Agents is an AI data-science team: specialised agents collaborate to
turn a tabular dataset and a business request into analysis, executable
Python, and stakeholder-facing recommendations.

The intended product is a prototype for structured/tabular data, not a
production AutoML platform. Do not claim that it supports unstructured data,
real-time streaming, enterprise deployment, or proprietary-system integration.

## Verified implementation

- `main.py` is a Streamlit interface. It collects an OpenAI API key, one or
  more uploaded datasets, and a user requirement before starting analysis.
- `agents/group_chat.py` orchestrates four AG2/AutoGen agents:
  `BusinessAnalyst`, `DataScientist`, `Coder`, and `BusinessTranslator`.
- The `Coder` agent generates Python and executes it through a Jupyter code
  executor. Treat uploaded data and generated code as untrusted; never use
  private or sensitive data for a demo run.
- The repository contains public/Kaggle-style tabular datasets, including
  house prices, bank churn, academic success, Titanic, telecom, and others.
- The README's house-price recommendation is an illustrative example. It is
  not a recorded benchmark result.

## Evaluation evidence

Project planning documents describe targets such as predictive performance
within 3% of a manual baseline, a 40% development-time reduction, and >92%
workflow reliability. Those are proposed success criteria, not results. Do not
publish them as achieved metrics unless a dated evaluation artifact proves them.

For a reproducible rerun, record the dataset, prompt, agent trace, generated
code, recommendation, runtime, and failures.

## Current repository state — preserve before changing

This worktree is intentionally dirty. Do **not** use `git reset`, `git
checkout --`, bulk deletion, or automated formatting without explicit user
approval.

The current state has an incomplete rename:

- Tracked source imports `agents`, but the working directory currently
  contains `multi-agents/`.
- The package marker is currently `multi-agents/init.py`, not
  `agents/__init__.py`.
- Several dataset directories were renamed from underscore-separated names to
  hyphen-separated names and are uncommitted.

Therefore the current worktree will fail to import `agents` after
dependencies are installed. Before running the app, choose one route:

1. Restore/run a clean `HEAD` baseline in a separate temporary worktree; or
2. Complete the current rename refactor consistently, with tests.

Never silently choose or mix these routes.

## Environment and safe rerun procedure

- Required Python: `>=3.12,<3.14` in `pyproject.toml`.
- `python3.12`, `uv`, and `poetry` are available on this machine. The default
  `/usr/bin/python3` is 3.9 and is unsuitable.
- The current shell has no installed `streamlit` package, so use an isolated
  Python 3.12 environment rather than installing dependencies globally.
- `pyproject.toml` declares AG2 0.9.9, Streamlit, OpenAI, and scikit-learn;
  `requirements.txt` is older and differs. Prefer the lockfile/`pyproject.toml`
  once the source-layout choice above is resolved.
- Never write an API key to this repository, source code, logs, screenshots,
  commits, or agent messages. Enter it only locally in the UI or through a
  temporary environment variable.

After the layout is runnable, use one non-sensitive sample dataset and capture
the evidence listed in **Evidence limits**. Stop if generated code attempts an
unexpected file, network, or destructive operation.

## Change discipline

1. Inspect `git status --short` and the relevant diff before editing.
2. Reproduce a failure before proposing a fix.
3. Make one minimal change at a time, with a focused test or smoke check.
4. Keep documentation aligned with code: the project has used both
   `GPT-4o-mini` and `gpt-4.1-mini` in different places; verify the active
   model before publishing a claim.
