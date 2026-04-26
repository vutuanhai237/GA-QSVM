# Runtime-First Refactor Design

Date: 2026-04-26
Scope: GA-QSVM runtime cleanup with breaking changes
Status: Draft for review

## Goal

Refactor this repository into a runtime-oriented Python codebase with clear boundaries between:

- runnable GA-QSVM application code
- reusable local optimization library code
- non-runtime research artifacts

This refactor is intentionally breaking. Backward compatibility for old script paths, import paths, and benchmark workflows is not a goal.

## Non-Goals

- Deep internal redesign of `qoop/`
- Preserving old benchmark, plotting, notebook, or one-off experiment workflows
- Keeping legacy root-level entrypoint behavior
- Supporting optional GPU, cloud, or benchmark-only dependencies in the main runtime path

## Constraints

- `qoop/` stays in-repo and is treated as a mostly stable local library
- Runtime behavior should be preserved where practical for the main GA-QSVM training and evaluation flows
- Refactor should make the main path testable without forcing benchmark dependencies into the default environment

## Recommended End State

The repository should move toward this structure:

```text
ga_qsvm/
  cli/
    train.py
    eval.py
  datasets/
    __init__.py
    registry.py
    split.py
  runners/
    train.py
    eval.py
  search/
    __init__.py
    space.py
  tracking/
    __init__.py
    wandb.py
  utils/
    __init__.py
    combinatorics.py

qoop/
tests/
docs/
requirements.txt
environment.yml
README.md
```

The root should stop being the place where orchestration logic lives. Runtime logic should live in `ga_qsvm/`, and CLI modules should be thin wrappers over runner APIs.

## Directory Decisions

### Keep

- `qoop/` as the local GA/quantum engine
- `README.md`, `requirements.txt`, `environment.yml`
- `images/` only if still referenced by docs

### Introduce

- `ga_qsvm/` as the application package
- `tests/` for runtime-focused automated verification
- `docs/superpowers/specs/` for design and planning artifacts

### Remove

- `experiment/`
- `notebook/`
- `bash/`

These directories are being treated as research artifacts, not production runtime surfaces. The user explicitly approved deleting them rather than isolating them.

## Architecture

### 1. CLI Layer

Responsibility:

- parse command-line arguments
- validate user-facing options
- call runner functions
- avoid owning orchestration logic

Modules:

- `ga_qsvm.cli.train`
- `ga_qsvm.cli.eval`

Design rules:

- no global `parse_args()` execution at import time
- expose `main(argv: list[str] | None = None) -> int`
- make argument defaults explicit and testable

### 2. Dataset Layer

Responsibility:

- dataset registry
- data loading
- train/test split
- scaling and PCA
- parameter-consistent preprocessing

Modules:

- `ga_qsvm.datasets.registry`
- `ga_qsvm.datasets.split`

Design rules:

- remove hard-coded train/test sizes where public parameters exist
- standardize function signatures across wine, digits, and cancer datasets
- keep dataset selection explicit through a registry rather than ad hoc dictionaries inside scripts

### 3. Search Layer

Responsibility:

- build hyperparameter search spaces
- enumerate RX/RY/RZ combinations
- create parameter combinations independent of the runner

Modules:

- `ga_qsvm.search.space`
- `ga_qsvm.utils.combinatorics`

Design rules:

- move `find_permutations_sum_n()` out of root-level utility usage into package code
- keep search-space construction pure and unit-testable

### 4. Runner Layer

Responsibility:

- orchestrate training and evaluation flows
- assemble dataset context
- build `MetadataSynthesis`
- configure and invoke `qoop.evolution.EEnvironment`
- manage validation/evaluation split policies

Modules:

- `ga_qsvm.runners.train`
- `ga_qsvm.runners.eval`

Design rules:

- move orchestration out of `main.py` and `eval.py`
- avoid hidden global state for fitness functions
- prefer explicit context objects or closures built inside runner functions

### 5. Tracking Layer

Responsibility:

- create W&B config payloads
- isolate runtime logging config from business logic

Modules:

- `ga_qsvm.tracking.wandb`

Design rules:

- W&B naming/config assembly should not live inline inside the main orchestration loop
- keep W&B optional at the call boundary where feasible, but do not redesign `qoop` logging internals in this phase

## `qoop/` Boundary

`qoop/` remains a local implementation dependency, not the refactor target.

Allowed changes inside `qoop/`:

- import fixes required for runtime correctness
- narrow compatibility fixes needed by the new runtime package

Disallowed changes in this phase:

- broad internal architecture cleanup
- packaging overhaul of `qoop/`
- deep algorithm changes

This keeps the blast radius focused on runtime cleanup rather than library reinvention.

## Dependency Strategy

The default runtime environment should represent the train/eval path only.

Required actions:

- reconcile `requirements.txt` with the actual runtime path
- keep `environment.yml` aligned enough for local parity
- remove benchmark-only dependencies from the default runtime surface where possible

Dependency classes:

- core runtime: `numpy`, `scikit-learn`, `qiskit`, `qiskit-machine-learning`, `wandb`
- optional/non-runtime: PennyLane GPU, ThunderSVM, CuPy, Covalent Cloud, TensorFlow paths not needed by the primary runtime

Specific direction:

- if MNIST is not part of the supported runtime path after cleanup, its TensorFlow dependency should not remain a default dependency without justification
- stale package metadata under `qoop/` can remain temporarily if it no longer controls the primary runtime path

## Deletion Policy

The following legacy research surfaces should be removed in this refactor:

- benchmark scripts under `experiment/`
- notebooks under `notebook/`
- shell wrappers under `bash/`

Reason:

- they are outside the approved runtime-first scope
- they pull in optional dependencies and outdated execution assumptions
- they make the repository structure noisy and blur the supported execution path

## Testing Strategy

Minimum required automated coverage for this refactor:

- dataset preprocessing tests
- combinatorics/search-space tests
- CLI argument parsing smoke tests
- runner smoke tests for orchestration setup without requiring full benchmark workflows

Recommended initial targets:

- verify `training_size`, `test_size`, `n_features`, and dataset choice are respected
- verify RX/RY/RZ permutation generation is stable
- verify CLI modules can parse arguments and invoke runner entrypoints
- verify train/eval runners can assemble environment configuration deterministically

Not required in this phase:

- full quantum algorithm correctness tests
- benchmark parity tests
- notebook execution tests

## README and User-Facing Contract

After refactor, the README should describe only the supported runtime path.

It should stop advertising deleted benchmark scripts, notebook workflows, or stale shell wrappers.

Expected user-facing changes:

- new invocation path for train/eval commands
- reduced set of supported workflows
- clearer distinction between runtime code and local engine code

## Migration Impact

This refactor intentionally breaks:

- old root-level script workflows
- old benchmark and notebook paths
- any external automation depending on `experiment/`, `notebook/`, or `bash/`
- imports that depend on current root-script layout

This refactor should preserve, after migration:

- the ability to run GA-QSVM training from a supported CLI
- the ability to run evaluation from a supported CLI
- the ability to reuse `qoop/` from the new runtime package

## Implementation Plan Shape

Implementation should proceed in four stages:

1. Create the new runtime package skeleton and tests scaffold
2. Move train/eval orchestration and dataset logic into `ga_qsvm/`
3. Replace root-level runtime entrypoints and remove research artifact directories
4. Normalize runtime dependencies and update README/documentation

## Risks

- `qoop.evolution.EEnvironment` currently mixes GA logic with W&B, persistence, and multiprocessing; the new runner must adapt to that without broad invasive changes
- deleting research artifacts removes recovery paths for old experiments
- missing tests in the current repo increase regression risk, so the refactor should add tests before broad movement is considered complete
- some dataset behavior may currently rely on accidental hard-coded defaults; standardizing APIs may surface latent assumptions

## Acceptance Criteria

- runtime orchestration no longer lives in `main.py` or `eval.py`
- supported train/eval flows live under `ga_qsvm/`
- dataset APIs consistently honor their parameters
- `experiment/`, `notebook/`, and `bash/` are removed
- automated tests exist for runtime-critical surfaces
- README matches the new supported runtime path
- the refactor does not depend on a deep `qoop/` rewrite

## Open Decisions Resolved

- Refactor style: runtime-first
- Compatibility stance: breaking cleanup
- `qoop/` treatment: mostly leave it alone
- Research artifacts: delete rather than isolate
- Preferred implementation scope: runtime extraction plus dependency normalization
