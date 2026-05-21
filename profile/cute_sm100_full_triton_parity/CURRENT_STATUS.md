# CuTe sm100 current status

This file is the authoritative current-status summary for the
`cute_sm100_full_triton_parity` work. The longer `progress.md` file is a
chronological experiment log and contains obsolete intermediate measurements.

## Completion scope

Under the current agreed performance policy, the default auto-amax benchmark
matrix is complete:

- Non-pseudo rows require `>=1.2x` Triton.
- Pseudo rows require strict speedup over Triton.
- Provided-`x_amax` rows are measured separately and are not part of this
  completion scope.

## Latest default auto-amax benchmark

Source: `benchmark_current.json`

- Total rows: `476`
- Rows meeting the required target: `476/476`
- Rows meeting the old all-rows `>=1.2x` policy: `476/476`
- Non-pseudo rows meeting `>=1.2x`: `338/338`
- Pseudo rows strictly faster than Triton: `138/138`
- Rows below the required target: `0`

Fresh benchmark regeneration with strict pseudo `>1.0x` target on 2026-05-21:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python profile/cute_sm100_full_triton_parity/benchmark_current.py
476/476 workloads meet required target
```

## Latest executable support audit

Source: `executable_support_current.json`

- CuTe missing Triton-runnable rows: `0`
- CuTe predicate failures: `0`
- Triton predicate/runnable rows: `423/405`
- CuTe predicate/runnable rows: `411/411`

The 18 Triton predicate-only failures are rows where Triton claims support but
the actual sm100 Triton kernel does not compile/run in the executable audit.

Fresh executable-support verification on 2026-05-21:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python profile/cute_sm100_full_triton_parity/audit_executable_support.py
triton_predicate 423 triton_runnable 405 cute_predicate 411 cute_runnable 411
triton_predicate_failures 18
cute_missing_runnable 0
cute_predicate_failures 0
```

## Validation

Fresh verification on 2026-05-21:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_quantize.py -q -rxXs -k cute_sm100
488 passed, 10 skipped, 34631 deselected, 1 warning in 36.55s
```

Latest recorded full `cute_sm100` quantize test selection:

```text
488 passed, 10 skipped, 34631 deselected, 1 warning
```

Latest recorded targeted provided-`x_amax` test selection:

```text
11 passed, 35118 deselected, 1 warning
```
