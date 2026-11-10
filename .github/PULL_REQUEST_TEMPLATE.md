## What and why

What does this change, and why? Link any issue it closes
(`Closes #123`).

## Type of change

- [ ] Bug fix
- [ ] New feature (guard, adapter, source, CLI command, capability)
- [ ] Documentation
- [ ] Refactor or performance
- [ ] CI or tooling

## Checklist

- [ ] `ruff check src tests` is clean.
- [ ] `pytest` passes (note any deselected slow test below).
- [ ] New behaviour has tests, including the fail-open path for guards.
- [ ] Public API or `GuardDecision` changes are in the README and `CHANGELOG.md`.
- [ ] No em dashes or en dashes.
- [ ] No secrets, tokens, or real customer data in fixtures, tests, or logs.
- [ ] Commit messages follow `type(scope): subject`.

## Notes for reviewers

Anything worth calling out: a design trade-off, a deselected test, a
follow-up left for later.
