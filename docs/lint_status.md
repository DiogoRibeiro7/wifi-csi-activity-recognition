# Lint, format and type-check status

The `Lint and format` CI job is **blocking**. The `Type check` job is not, and
this page records why and what would clear it.

## Blocking today

`pre-commit run --all-files` exits 0. Every hook below gates merges:

| Hook | Role |
|---|---|
| black | formatting |
| isort | import ordering |
| flake8 | code issues (`--min-python-version=3.10`) |
| pydocstyle | docstrings, google convention |
| bandit | security, scoped to the package |
| trailing-whitespace, end-of-file-fixer | whitespace |
| check-yaml / toml / json, merge conflicts, large files, debug statements, docstring-first | file hygiene |
| nbqa-black, nbqa-isort | notebooks |

## What the first clean run required

The hooks had never run against this tree — the config file was named
`pre-commit-config.yml`, which is not the path pre-commit reads. Most of what
surfaced was misconfiguration rather than bad code.

**Every hook revision was pinned to mid-2023.** flake8 6.0.0 ships a
pycodestyle that predates PEP 701 f-string tokenization, so on modern Python it
read the `:` in `f"{ratio:.1f}"` as a syntax colon and the `;` inside an
f-string as a statement separator — 20 × E231 and 3 × E702, all false. Updating
the revisions removed them without touching a line of code.

**Two docstring linters disagreed.** `flake8-docstrings` (pep257 defaults) and
`pydocstyle` (`--convention=google`) both ran. They enforce mutually exclusive
rules — D212 against D213 among others — so some docstrings could not satisfy
both. `flake8-docstrings` was removed; pydocstyle is the single authority.

**flake8-typing-imports targeted Python 3.5.** Its default floor demanded
`TYPE_CHECKING` guards for typing constructs available since long before this
project's floor of 3.10. Fixed with `--min-python-version=3.10`.

**bandit scanned build artefacts.** With `pass_filenames: false` the hook's
`exclude:` never reached bandit, so `-r .` walked `build/` — a gitignored copy
of the package — and reported everything twice. Now scoped to the package.

**nbqa pinned older formatters** than the main hooks (black 23.7.0 vs 26.5.1),
so notebooks would have been formatted to a different standard than modules.

Genuine code fixes were smaller: 43 unused imports, 4 over-length lines, one
ambiguous variable name (`l`), one misplaced import, 18 docstring summaries
starting on the wrong line, 4 missing docstrings, and 6 silent
`except Exception: pass` blocks in the streaming pipeline that now log at debug
instead of swallowing the cause.

## Not blocking: type checking

`mypy wifi_activity_recognition --ignore-missing-imports --no-strict-optional`,
run with the project's dependencies installed:

```
Found 245 errors in 49 files (checked 78 source files)
```

By category:

| Count | Code | Nature |
|---|---|---|
| 64 | `unused-ignore` | `# type: ignore` comments no longer needed |
| 48 | `no-any-return` | returning `Any` from a typed signature |
| 41 | `arg-type` | genuine argument type mismatches |
| 37 | `no-untyped-def` | missing annotations |
| 11 | `name-defined` | undefined names in annotations |
| 44 | assorted | `assignment`, `index`, `union-attr`, `var-annotated`, … |

These are real, not artefacts — the run above had numpy, torch and click
present. Clearing them is annotation work across most of the package, which is
why it is tracked rather than bundled into a formatting pass.

**The type-check job must run with dependencies installed.** A bare pre-commit
hook environment has none, so third-party calls resolve to `Any` and mypy
reports 140 phantom `untyped-decorator` and `unused-ignore` errors that say
nothing about the code. The mypy hook in `.pre-commit-config.yaml` is therefore
marked `stages: [manual]`; CI runs mypy directly instead.

**Scope is the package, not the tests.** Type-checking the suite under
`disallow_untyped_defs` produced 336 further errors, almost all demands to
annotate test functions. That is not where the value is.

### Suggested order

1. `unused-ignore` (64) — deletions, no behaviour change, immediate win
2. `no-untyped-def` (37) — mechanical signatures
3. `arg-type` (41) and `name-defined` (11) — the ones most likely to be real bugs
4. `no-any-return` (48) — often needs a genuine decision about the return type

Flip `types` to blocking, and remove `stages: [manual]` from the mypy hook,
once the count reaches zero.
