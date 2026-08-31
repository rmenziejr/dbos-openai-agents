# PostgreSQL and SandboxAgent Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a Docker Compose PostgreSQL environment and one runnable notebook showing regular, sandboxed, and agent-as-tool OpenAI Agents SDK patterns.

**Architecture:** PostgreSQL runs locally in Compose and supplies DBOS persistence for the regular and coordinator examples. The notebook runs DBOS-backed examples through `DBOSRunner`, while the official `SandboxAgent` example runs through the upstream `Runner` with a local sandbox client and an explicitly mounted temporary directory.

**Tech Stack:** Docker Compose, PostgreSQL 16, Python 3.10+, DBOS, OpenAI Agents SDK 0.14+, Jupyter/nbformat.

**Spec:** `docs/superpowers/specs/2026-08-31-postgres-sandbox-agent-notebook-design.md`

## Global Constraints

- Set the minimum `openai-agents` version to `0.14.0`.
- Keep PostgreSQL credentials local-only and overridable through Compose environment variables.
- Do not place API keys or database credentials in the sandbox manifest or notebook output.
- Use `UnixLocalSandboxClient` and an explicitly mounted temporary data directory for `SandboxAgent`.
- Document that `SandboxAgent` uses upstream `Runner`, not `DBOSRunner`.

---

## File Structure

- `pyproject.toml`: declared SDK version floor and notebook development dependencies.
- `uv.lock`: resolved dependency graph for the new SDK and notebook tooling.
- `docker-compose.yml`: local PostgreSQL service, health check, and persistent named volume.
- `README.md`: startup command and notebook prerequisites.
- `notebooks/durable_agents_examples.ipynb`: executable examples and environment checks.
- `tests/test_sandbox_agent_dependency.py`: imports the required sandbox API from the installed dependency.
- `tests/test_notebook_examples.py`: validates the notebook and its required example sections.

### Task 1: Upgrade SDK and add notebook tooling

**Files:**
- Modify: `pyproject.toml:22-36`
- Modify: `uv.lock`
- Create: `tests/test_sandbox_agent_dependency.py`

**Interfaces:**
- Consumes: the `openai-agents` dependency declared by `pyproject.toml`.
- Produces: importable `agents.sandbox.SandboxAgent` for the notebook.

- [ ] **Step 1: Write the failing dependency-capability test**

```python
from agents.sandbox import SandboxAgent


def test_agents_sdk_exposes_sandbox_agent() -> None:
    assert SandboxAgent.__name__ == "SandboxAgent"
```

- [ ] **Step 2: Run the test to verify it fails with the current dependency**

Run: `uv run pytest tests/test_sandbox_agent_dependency.py -q`
Expected: collection fails because `agents.sandbox` is unavailable under the current lockfile.

- [ ] **Step 3: Raise the dependency floor and add notebook tooling**

```toml
[project]
dependencies = [
    "dbos>=2.10.0",
    "openai-agents>=0.14.0",
]

[dependency-groups]
dev = [
    # existing entries
    "jupyterlab>=4.0.0",
]
```

Run `uv lock` to update the lockfile. Do not add an API key or Docker dependency to Python requirements.

- [ ] **Step 4: Run the dependency test and type check**

Run: `uv run pytest tests/test_sandbox_agent_dependency.py -q && uv run mypy dbos_openai_agents tests`
Expected: both commands exit 0.

- [ ] **Step 5: Commit the dependency upgrade**

```bash
git add pyproject.toml uv.lock tests/test_sandbox_agent_dependency.py
git commit -m "build: require sandbox-capable agents sdk"
```

### Task 2: Add the local PostgreSQL Compose environment

**Files:**
- Create: `docker-compose.yml`
- Modify: `README.md:after the usage section`

**Interfaces:**
- Consumes: optional `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` environment variables.
- Produces: `postgresql://dbos:dbos@localhost:5432/dbos` when defaults are used; a `postgres` service with a passing `pg_isready` health check.

- [ ] **Step 1: Add the Compose model**

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-dbos}
      POSTGRES_USER: ${POSTGRES_USER:-dbos}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-dbos}
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 5s
      timeout: 5s
      retries: 10
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

- [ ] **Step 2: Document local startup and connection details**

Add a README section with `docker compose up -d`, `docker compose down`, the default database URL, and a statement that the values are local-development defaults.

- [ ] **Step 3: Validate the Compose model**

Run: `docker compose config`
Expected: exit 0 and a rendered `postgres` service containing the health check and named volume.

- [ ] **Step 4: Commit the local environment**

```bash
git add docker-compose.yml README.md
git commit -m "docs: add local postgres development environment"
```

### Task 3: Build and validate the executable notebook

**Files:**
- Create: `notebooks/durable_agents_examples.ipynb`
- Create: `tests/test_notebook_examples.py`

**Interfaces:**
- Consumes: `OPENAI_API_KEY`, the Compose PostgreSQL URL, `DBOSRunner`, `SandboxAgent`, and `Agent.as_tool()`.
- Produces: three labeled, executable notebook examples; regular and coordinator examples return DBOS workflow information.

- [ ] **Step 1: Write the failing notebook-structure test**

```python
import json
from pathlib import Path


NOTEBOOK = Path("notebooks/durable_agents_examples.ipynb")


def test_notebook_contains_all_agent_examples() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
    )
    assert "Regular durable agent" in source
    assert "SandboxAgent" in source
    assert "Agent as a tool" in source
    assert "DBOSRunner.run" in source
```

- [ ] **Step 2: Run the test to verify it fails because the notebook is absent**

Run: `uv run pytest tests/test_notebook_examples.py -q`
Expected: FAIL with `FileNotFoundError` for `notebooks/durable_agents_examples.ipynb`.

- [ ] **Step 3: Create the notebook with these ordered sections**

1. Markdown title, setup instructions, and the `docker compose up -d` command.
2. Imports and a prerequisite cell that rejects a missing `OPENAI_API_KEY` and checks PostgreSQL before `DBOS.launch()`.
3. A DBOS configuration cell using `postgresql://dbos:dbos@localhost:5432/dbos`.
4. A regular `Agent` plus a `@function_tool` and `@DBOS.step()` tool, invoked through a `@DBOS.workflow()` using `await DBOSRunner.run(...)`.
5. A `SandboxAgent` using `Manifest(entries={"data": LocalDir(...)})`, `UnixLocalSandboxClient`, and `await Runner.run(..., run_config=RunConfig(sandbox=SandboxRunConfig(...)))`.
6. A specialist `Agent`, `specialist.as_tool(...)`, and a DBOS-backed coordinator `Agent` invoked through `DBOSRunner.run(...)`.
7. Cleanup instructions for `DBOS.destroy()` and `docker compose down`.

Use `nbformat.v4` to generate valid notebook JSON with no stored outputs and no secrets.

- [ ] **Step 4: Run structural validation and the notebook test**

Run: `uv run python -c "import nbformat; nbformat.validate(nbformat.read(notebooks/durable_agents_examples.ipynb, as_version=4))" && uv run pytest tests/test_notebook_examples.py -q`
Expected: both commands exit 0.

- [ ] **Step 5: Commit the runnable examples**

```bash
git add notebooks/durable_agents_examples.ipynb tests/test_notebook_examples.py
git commit -m "docs: add durable agent notebook examples"
```

### Task 4: Final integration verification

**Files:**
- Verify: `pyproject.toml`, `docker-compose.yml`, `README.md`, `notebooks/durable_agents_examples.ipynb`, `tests/`

**Interfaces:**
- Consumes: all artifacts from Tasks 1-3.
- Produces: a verified local setup and notebook handoff.

- [ ] **Step 1: Validate dependency imports and notebook structure**

Run: `uv run pytest tests/test_sandbox_agent_dependency.py tests/test_notebook_examples.py -q`
Expected: both tests pass.

- [ ] **Step 2: Validate Compose and the complete regression suite**

Run: `docker compose config && uv run pytest -q`
Expected: Compose rendering succeeds and all tests pass.

- [ ] **Step 3: Validate static checks**

Run: `uv run mypy dbos_openai_agents tests && uv run black --check dbos_openai_agents tests && uv run isort --check-only dbos_openai_agents tests`
Expected: all commands exit 0. If pre-existing formatting failures occur in untouched files, report them separately without reformatting unrelated files.

- [ ] **Step 4: Commit any final validation-only fixes**

```bash
git add pyproject.toml uv.lock docker-compose.yml README.md notebooks/durable_agents_examples.ipynb tests/
git commit -m "test: verify postgres sandbox agent examples"
```
