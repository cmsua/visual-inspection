# Portable Python conventions

This is a reusable style guide, condensed from the **Additional Python
Conventions** section of `AGENTS.md`. It does not replace that file automatically.

To adopt it elsewhere, copy this file into that repository and instruct its
`AGENTS.md`: “Read and follow `docs/STYLES.md` before editing Python.”
Alternatively, copy only the **Core rules** section into `AGENTS.md`. Keep the
tooling instructions available separately. Declare the project's package roots,
Python version, artifact paths, and domain-specific exceptions locally. Examples
below assume a package named `src`; substitute the actual package name when needed.
Explicit project instructions take precedence. Never silently drop an exception
while shortening these rules.

## Core rules

### Formatting

- **F1 — Spacing and commas.** Use four spaces and ordinary token spacing; do not
  align equals signs. Omit optional trailing commas in definitions, calls,
  instantiations, collections, and imports, including multiline forms. Preserve
  syntax-required singleton-tuple commas: `(value,)` and `array[index,]`.
- **F2 — Selective wrapping.** No fixed line-length limit. Keep readable calls,
  messages, comparisons, comprehensions, shapes, coordinate tuples, and short
  matrix rows together. Wrap crowded expressions by logical structure; do not
  introduce temporary variables or recursively expand nested expressions just
  to shorten lines. Use actual parameter names as keywords when they clarify a
  crowded call. Expand substantial dictionaries by key/value entry; group short
  related list/tuple entries and give complex entries their own lines.
- **F3 — Definitions and calls.** A multiline definition or call keeps `(` with
  its name, puts each parameter/argument on a separate continuation line, and
  closes on a separate line. Include `self`, `cls`, class bases/keywords, and
  definition separators `*` and `/`; a tuple argument remains one argument.
  **Wrapper exception:** expand crowded inner content while keeping a simple
  outer wrapper compact, such as `Model(Config(` … `))`, `append({` … `})`, or
  `next((` … `), None)`. Readable nested calls need not expand.
- **F4 — Conditions and strings.** Keep simple conditions compact. When wrapped,
  use parentheses and break before `and`/`or`, retaining whole comparisons.
  Preserve parenthesized tensor comparisons with `&`/`|`. Wrap a crowded call
  before its conditional expression; a short `if … else …` suffix may follow
  the closing `)`. Split long messages into adjacent parenthesized string
  literals, preserving spaces and never inserting commas between fragments.
- **F5 — Chains.** Keep short chains inline. In multiline chains, keep `.method(`
  attached to its receiver or the preceding `)`, with one argument per line and
  a separate closing `)`. Apply this to the initial call too. Argument-free
  calls may stay inline; never place `.method` alone on a new line.
- **F6 — Block spacing.** Leave a blank line after a completed control-flow block
  before the next same-indent statement, including a fallback `return`. Related
  short guards/loops may be consecutive, but the final one still needs that
  separation. Never separate `elif`/`else`/`except`/`finally` from their block or
  add a trailing blank line just because a block ends its enclosing scope.

### Documentation, quotes, names, and types

- **D1 — Docstring scope.** Start files with imports after any required interpreter
  or encoding directive; omit module/file-header docstrings. Add no docstrings
  outside `src/`. Inside it, use NumPy-style docstrings for major reusable
  classes and APIs called directly by notebooks/scripts. Internal helpers,
  including nested helpers and single-underscore methods, use comments instead.
  A public-looking name or re-export does not establish public API status.
  Documented interface methods such as `__init__`, `__call__`, and `__getitem__`
  may retain docstrings.
- **D2 — Docstring layout.** Put opening and closing triple-double quotes on their
  own lines, even for a one-sentence description; omit trailing spaces. Start
  function bodies immediately after the closing delimiter; separate a class
  docstring from its attributes/methods with one blank line.
- **D3 — Docstring notation.** Write `name : Type`. Use annotation-style collection
  names (`List[Dict]`, `Tuple[Tensor, Tensor]`, `Sequence[str]`) and lowercase
  scalars. Give useful, accurate element types. In documentation, write
  `List[Tensor], optional`, `str or Path`, or
  `Callable or List[Callable], optional`, including return fields; actual code
  annotations retain `Optional[...]` and `Union[...]`.
- **D4 — Inline references.** Enclose shapes and variable references in single
  backticks in docstrings and user-facing messages, including interpolated
  shapes. Keep NumPy parameter/type declarations unquoted.
- **Q1 — Semantic quotes.** Single quotes for keys, internal identifiers, strict
  parameter values, configuration constants, file modes, and encodings; double
  quotes for errors, print/log messages, and other user-facing text.
- **T1 — Names and types.** Use descriptive `snake_case` functions/variables,
  `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Annotate parameters and
  returns, but not local variables inside functions. Prefer `typing` generics
  (`List`, `Dict`, `Tuple`, `Optional`) and import `Tensor` from `torch` instead
  of spelling annotations `torch.Tensor`.

### Imports and boundaries

- **I1 — Import ownership.** Across folders, import from the owning package's
  exports; within a folder, import the defining sibling filename. Direct imports
  from private files beginning `_` are allowed; a private symbol alone is not
  an exception. Define exports in `__init__.py`, where implementation imports
  are appropriate. Apply this to scripts, tests, notebooks, and examples too.
  These restrictions concern project modules, not third-party APIs. Do not rename
  files or add dynamic imports to evade them; keep low-level dependencies acyclic.
- **I2 — Import ordering.** Group standard library, third-party, PyTorch, then
  project/relative imports, separated by blank lines. Place `import` before
  `from` within groups; alphabetize modules ignoring case. Within each
  `from ... import ...` statement, use `CONSTANT -> ClassObject -> func`:
  constants, then classes, then functions, alphabetically within each category
  ignoring case. Apply this to package exports and private symbols too;
  `PreflightSession` precedes `_preflight_result`. Resolve ambiguous names by
  the symbol's actual role.
  Order relatives closest-to-furthest. Document concrete initialization/circular
  dependency exceptions beside affected imports; preserve necessary deferred,
  type-checking, and backend-selection ordering.
- **I3 — Import spelling.** Prefer relative imports inside `src/`, absolute
  `src...` imports in tests/scripts. Do not mutate `sys.path`. Keep short imports
  compact; wrap crowded symbol lists one name per line, without a final comma.
- **O1 — Code organization.** Put execution-only CLI entry points in `scripts/`;
  never import them into application code. Group related tests by behavior,
  preserve coverage, and reuse private helpers instead of importing test modules.

### Documentation and experiment workflow

- **O2 — Work reports.** Final responses list every edited/added file plus changes
  and verification. Put durable rules/contracts/plans in `AGENTS.md`, onboarding
  and workflows in `README.md`, and detailed run results in artifacts. Do not
  append per-run diagnostics or test counts to `AGENTS.md`; record an evaluation
  milestone only after its required evaluation and artifact checks are complete.
- **A1 — Paths and lifecycle.** Use repository-relative `outputs/` for figures,
  tables, inspection results, and final reports; `logs/` for training logs,
  checkpoints, weights, resolved run configs/manifests, environment records, and
  source provenance. Match experiment/run IDs across both and record the output
  path in run metadata. Resume from `logs/`; preserve immutable snapshots.
  Create output directories only when writing, not when initializing trainers;
  tests write figures to temporary paths.
- **A2 — Configuration.** Put reusable YAML in `configs/`, optionally grouped by
  experiment, with named sections such as `data`, `model`, `loss`, and `train`.
  Scripts/jobs consume YAML; notebooks declare explicit Config values in cells
  and save resolved snapshots through the trainer. JSON suits domain manifests
  and resolved snapshots. Never put shared input configs in a run folder or
  `logs/setup`. Project-specific data/annotation paths remain local contracts.
- **V1 — Plot lifecycle.** Public plot helpers default to `plt.show()`, then close
  and return `None`. Accept `savefig=None`; a supplied `str`/`Path` saves PNG at
  300 DPI before display and creates parents as needed. For `show=False`, return
  the figure or named figure dictionary; callers own closing it. Composed helpers
  must not show incomplete nested figures. Document multi-figure filename rules.
- **V2 — Plot layout.** Leave room for headings and explicit vertical space
  between image rows so titles cannot overlap images. Preserve image aspect
  ratios; use equal axis scales where coordinates represent equal distances.
- **B1 — Notebook behavior.** Use the project's configured dimensions and splits;
  never silently shrink them for a demonstration. Synchronize Markdown and code.
  Prefer a named result/DataFrame/Series as the final expression, placing useful
  independent outputs in separate cells. Otherwise use `print()` for labels,
  multiple scalars, and live progress; do not invent dictionaries to avoid it.
  Call plotting helpers directly without duplicate display/save/close operations.
  Follow A1 paths and never execute notebooks merely to edit or statically check them.
- **X1 — Tests.** Run pytest from the repository root with `PYTHONPATH=.`.

### Examples of the less conventional rules

```python
# Inside src/engine/adapter.py, importing its sibling trainer.py:
from .trainer import Trainer

# From another folder:
from src.engine import Trainer

model = Model(Config(
    channels=channels,
    image_size=(height, width)
))

records.append({
    'name': name,
    'shape': (height, width)
})

message = (
    "The saved labels need reconciliation. "
    "Review the preview before applying the assignment."
) if labels_exist else "No saved labels require reconciliation."
```

## Ruff: enforce the objective subset

Use **`ruff check`**, with a reviewed selection of lint rules. Do not use
`ruff format` as the authority for this style: its wrapping, quote policy, and
comma insertion can conflict with the core rules. A clean lint run establishes
only the selected checks, not complete compliance.

The commands/configuration below were checked with Ruff **0.16.9**. Pin the
reviewed version in each project's development tooling and recheck fixtures
before upgrading. Install into the intended environment:

```bash
python -m pip install 'ruff==0.16.9'
python -m ruff --version
```

### Standalone configuration

Save this as `ruff.toml` in the destination project. This guide does not create
that file or alter an existing configuration. A standalone Ruff config needs no
packaging/build metadata. If using `pyproject.toml`, prefix these tables with
`tool.ruff` instead; do not define competing configurations unintentionally.
Ruff supports both forms; explicit `--config` makes the chosen file clear.
[Configuration reference](https://docs.astral.sh/ruff/configuration/)

```toml
# Adapt the Python version and excluded artifact directories to the project.
target-version = 'py312'
indent-width = 4
extend-exclude = ['venv', 'data', 'logs', 'outputs']

[lint]
# Several conventional whitespace checks are preview rules in this pinned release.
preview = true
explicit-preview-rules = true
select = [
    'E4', 'E7', 'E9', 'F',
    'E101', 'E111', 'E114', 'E117',
    'E201', 'E202', 'E211', 'E221', 'E222', 'E225', 'E231',
    'E241', 'E242', 'E251', 'E271', 'E272', 'E273', 'E274', 'E275',
    'W291', 'W292', 'W293',
    'COM819',
    'N801', 'N802', 'N803', 'N806',
    'ANN001', 'ANN002', 'ANN003',
    'ANN201', 'ANN202', 'ANN204', 'ANN205', 'ANN206',
    'D201', 'D202', 'D204', 'D209', 'D213', 'D300'
]

[lint.per-file-ignores]
# Public re-exports are intentional. Do not disable unused-import checks globally.
'**/__init__.py' = ['F401']
```

No `E501` means no lint-enforced line-length cap. The selected spacing/indentation
rules check conventional whitespace; the selected `N` rules cover class,
function, argument, and function-local naming, not semantic constant detection.
`ANN` checks missing annotations, not type correctness or the desired generic
spelling. Implicit `self`/`cls` are not required to have explicit annotations by
these checks. The selected `D` rules check existing docstrings only and do not
require new docstrings everywhere. They are partial layout checks: helper scope,
NumPy field notation, single-backtick references, and even every delimiter-only
case still require review/custom checks. Do not enable the whole `D` family:
rules requiring module docstrings or single-line summaries conflict with D1/D2.
Preview is enabled only to activate explicitly selected preview rules such as
the whitespace checks; `explicit-preview-rules` prevents broad prefixes from
silently opting into other preview rules. Keep the version pinned and revalidate
on upgrades.

The baseline correctness rules can report intentional notebook/deferred import
placement (`E402`). Where necessary, use a narrow, explained exception; preserve
the actual initialization dependency rather than moving imports mechanically.

### Check, preview, and fix

```bash
# Read-only diagnostics, including notebooks discovered by Ruff:
python -m ruff check --config ruff.toml .

# Preview available safe fixes before applying them:
python -m ruff check --config ruff.toml --diff .

# Apply reviewed safe fixes, inspect the diff, then check again:
python -m ruff check --config ruff.toml --fix .
git diff --check
git diff
python -m ruff check --config ruff.toml .

# Run relevant tests; the suite may need project-specific environment settings:
PYTHONPATH=. python -m pytest -q
```

Use the read-only command as a CI gate after the project meets the chosen baseline.
Do not claim a new profile passes an existing project until actually checking it.
Ruff respects discovery/exclusion settings; Markdown code blocks need separate
extraction/checking. Do not execute notebooks to lint them.
[File discovery and notebooks](https://docs.astral.sh/ruff/configuration/#jupyter-notebook-discovery)

### Important gaps and conflicting automation

| Preference | Enforcement and limitation |
| --- | --- |
| No unnecessary trailing commas | `COM819` catches some cases, **not multiline trailing commas** in the tested version. A local probe with `call(` / `value,` / `)` passed. Add a token/AST checker covering calls, definitions, imports, all collections and tuple subscripts; preserve singleton tuples and test starred tuple forms. |
| Direct sibling imports; package exports across folders | Requires repository-aware import resolution. Check deferred/type-checking imports too; permit private implementation modules and `__init__.py` exports. Ruff import sorting does not establish this rule. |
| No annotated locals; preferred `typing` and `Tensor` spelling | Requires AST checks/review. Avoid `UP006`, `UP007`, and `UP045`, which push alternative annotation spellings. |
| Semantic single/double quotes | Requires context/review. Do not enable a blanket `Q000` quote policy or globally normalize strings. |
| Selective wrapping, chains, post-block spacing | Review/custom checks must preserve wrapper, guard-group, clause, and scope-end exceptions. A fixed-width formatter cannot express this style. |
| Docstring locations and exact notation | Check allowed API scope and project roots; Ruff's selected layout checks are only partial. Do not enable `D200`, `D203`, `D205`, or `D212` without reviewing conflicts with the permitted layouts. |
| Project organization, plot lifecycle, experiments, notebooks | Review and behavior tests; these are not generic lint properties. |

`COM819`'s published description is broader than the multiline behavior observed
locally. Treat the fixture result as the coverage boundary. Also avoid `COM812`
and `COM818`, which demand commas in forms this style permits without them.
`skip-magic-trailing-comma` is not an “omit all trailing commas” switch.
[COM819 documentation](https://docs.astral.sh/ruff/rules/prohibited-trailing-comma/)

### Import sorting: optional review aid

`I001` can help sort imports, but is intentionally absent from the baseline:
Ruff's import rewriter also chooses wrapping and may introduce trailing commas.
It cannot validate sibling/package ownership or decide initialization exceptions.
If testing it, add these settings to `ruff.toml` and run the preview command below;
do not treat its rewrite as authoritative for F1–F3.

```toml
[lint.isort]
known-first-party = ['src'] # Adapt to the real importable package names.
section-order = ['future', 'standard-library', 'third-party', 'torch', 'first-party', 'local-folder']
case-sensitive = false
order-by-type = true
from-first = false
relative-imports-order = 'closest-to-furthest'
split-on-trailing-comma = false

[lint.isort.sections]
torch = ['torch', 'torchvision']
```

```bash
python -m ruff check --config ruff.toml --select I001 --diff .
```

Keep required `__future__` imports first. The custom PyTorch section, case-insensitive
sorting, constants/classes/functions symbol grouping, and relative-depth setting approximate
I2; inspect any proposed wrapping or dependency reordering. Ruff infers symbol
categories from spelling, so review ambiguous names against their actual roles. Document narrowly scoped
`# isort: skip` / `# isort: off` exceptions if adopting this check around necessary
initialization ordering. [Import settings](https://docs.astral.sh/ruff/settings/#lintisort)

Custom checks described here are a proposed enforcement layer, not tools supplied
by this document. Before accepting one, test compliant and violating examples,
including notebooks and required singleton commas. Until then, review those gaps
explicitly and report the exact checks run.
