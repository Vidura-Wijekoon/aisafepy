# Contributing to AIsafePy

Thanks for helping. AIsafePy is a security-adjacent library, so the bar
for correctness and clarity is high: a guard that fails in a surprising
way is worse than no guard. This guide covers setup, the quality bar,
and the three extension points people reach for most.

## Development setup

With [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv
uv pip install -e ".[dev,all]"
uv run pytest
```

With plain pip:

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"
pytest
```

The `all` extra pulls in the optional ML and agent-framework
dependencies. For a fast loop you can install just `".[dev]"`; the tests
that need heavy dependencies either skip or fail open, except the single
model-download test described below.

## Running tests

```bash
pytest                      # the whole suite
pytest tests/test_cli.py    # one module
pytest -m "not slow"        # skip slow tests
```

One test, `tests/test_adapt_sources.py::test_cluster_failures_groups_similar_inputs`,
downloads a sentence-transformers model on first run. To skip it:

```bash
pytest --deselect tests/test_adapt_sources.py::test_cluster_failures_groups_similar_inputs
```

The offline benchmarks are smoke-tested as part of the suite
(`tests/test_benchmarks_smoke.py`) and never touch the network.

## Linting and types

```bash
ruff check src tests
ruff format src tests       # apply formatting
mypy src/aisafepy           # type check (non-strict)
```

CI runs `ruff check` and `pytest` on Python 3.10, 3.11, and 3.12. Keep
both green. The repository convention is no em dashes or en dashes in
code, comments, or docs; use hyphens or restructure the sentence.

## Commit messages

Follow the `type(scope): subject` convention visible in `git log`:

```
feat(stream): add a sliding-window regex guard
fix(flow): propagate capabilities through Tainted.__getitem__
docs(adapt): clarify the canary rollback signal
chore: bump ruff and retune lint config
```

Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `perf`,
`chore`, `ci`, and `security`. Keep the subject in the imperative mood
and under about 70 characters.

## Adding a guard

A guard is any object matching the `Guard` protocol in
`aisafepy.stream.pipeline`: it has a `name` and `tier` and an
`async def __call__(self, ctx: Context) -> GuardDecision`.

1. Put deterministic guards in `aisafepy/stream/deterministic.py` and
   model-backed guards in `classifiers.py`, `forecasters.py`, or
   `probes.py`. Thin wrappers around third-party tools go in
   `aisafepy/contrib/`.
2. **Fail open with evidence.** If a dependency is missing or the guard
   raises, return `GuardDecision.allow(...)` with an `evidence` entry
   explaining why, never a traceback. The pipeline already wraps guard
   calls this way; your guard should do the same for its own internal
   failures (missing model, timeout, bad input).
3. Enforce an input cap. Long inputs are a denial-of-service vector;
   follow `RegexGuard.max_input_chars`.
4. Attach useful `evidence`: matched spans, probabilities, thresholds.
   This is what an operator reads at 3 a.m.
5. Add unit tests that cover allow, the terminal action, and the
   fail-open path. Use `StubClassifierGuard` as a model-free stand-in so
   tests stay offline.
6. Export it from the relevant `__init__.py` and mention it in the
   README quickstart if it is a headline capability.

## Adding an adapter

Adapters wrap an agent framework so IFC enforcement happens at the
tool-call boundary. See `aisafepy/flow/adapters/` for the five existing
ones.

1. Add a module under `aisafepy/flow/adapters/` with a
   `secure_<framework>(agent, policy, **kwargs)` entry point.
2. Import the framework lazily inside the function, not at module load,
   so `from aisafepy.flow import Policy` stays cheap.
3. Promote bare-string tool arguments from the model to `Tainted` with
   `integrity="UNTRUSTED"` before calling `evaluate_call`; model output
   is never trusted.
4. Translate `IFCViolationError` into the framework's native safety
   signal (a tripwire, an exception, a mediation callback).
5. Register the framework name in `aisafepy/flow/__init__.py`'s
   `_infer_framework` and `secure_agent` dispatch.
6. Add an integration test marked `@pytest.mark.integration` that uses a
   minimal fake agent object, so it runs without the real framework.

## Adding a source

Sources normalise a red-team or production format into
`FailureRecord`. See `aisafepy/adapt/sources.py`.

1. Add a class whose `__iter__` yields `FailureRecord`s. Keep it to the
   roughly 30-line shim shape the existing adapters use.
2. Duck-type external clients rather than importing them, so the source
   can be constructed without the upstream package installed.
3. Normalise defensively: tolerate schema drift across versions and
   coerce scores with the `_truthy` / `_as_float` helpers.
4. Export it from `aisafepy/adapt/__init__.py` and add a test that reads
   a small fixture file.

## Pull request checklist

- [ ] `ruff check src tests` is clean.
- [ ] `pytest` passes (note any deselected slow test in the PR body).
- [ ] New behaviour has tests, including the fail-open path for guards.
- [ ] Public API changes are reflected in the README and `CHANGELOG.md`.
- [ ] No em dashes or en dashes.
- [ ] No secrets, tokens, or real customer data in fixtures or logs.
- [ ] Commit messages follow `type(scope): subject`.

## Reporting security issues

Do not open a public issue for a vulnerability. Follow `SECURITY.md`.
