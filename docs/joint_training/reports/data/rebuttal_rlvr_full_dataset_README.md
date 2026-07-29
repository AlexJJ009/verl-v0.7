# RLVR Full Private Handoff Candidate (2026-07-29)

This directory contains the complete mathematical and code-task dataset bundle
currently selected for the Meituan handoff. It includes third-party benchmark
prompts/tests whose public redistribution rights have not been established, so
the complete bundle is **private-handoff only**. Do not make a repository that
contains this complete bundle public.

The repository-root `data/` tree is the canonical handoff mapping. Download the
repository directly to `DATASET_ROOT`; training and evaluation paths then live
under `${DATASET_ROOT}/data/...`. The inventory separately records every real
source path below `/data-1/dataset` (which resolves to `/data-2/dataset`) so the
handoff layout does not obscure local provenance.

## Included payload

- Mathematical training: the 7,500-row RL-format training parquet.
- Mathematical evaluation: the current rebuttal RLVR Math-7 inputs (AIME-2025,
  MATH-500, AMC23, AQUA, GSM8K, MAWPS, and SVAMP).
- Code training: the 10,000-row KodCode author-signature-v2 RL parquet.
- Code evaluation: the executable official Code-4 family used by this project:
  HumanEval+, MBPP+, LiveCodeBench release_v5, and BigCodeBench.

There is no executable Code-7 dataset contract in the project. EvalPlus reports
base and plus metrics from the same HumanEval+/MBPP+ inputs; those metrics are
not additional dataset files.

## Layout

```text
data/
  math/train_rl_format.parquet
  math7/
    AIME-2025/
    MATH-500/
    AMC23/
    AQUA/
    gsm8k/
    MAWPS/
    SVAMP/
  code/verl_rl/
metadata/
  publication_inventory.json
  checksums.sha256
```

## Integrity check

Run from this directory:

```bash
sha256sum -c metadata/checksums.sha256
```

`metadata/publication_inventory.json` records each payload file's source path,
row count, byte count, and SHA-256 digest.

The evaluator source revisions are fixed to:

- EvalPlus: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`
- BigCodeBench: `09dd993f46c3fbf3a799465bb96d524edcb0b199`
- LiveCodeBench: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`

## Explicit exclusion

The 3.6 GB LiveCodeBench release_v5 input/output SQLite index is an evaluator
asset, not a benchmark parquet, and is intentionally not included. A runtime
that scores LiveCodeBench must provision that evaluator asset separately and
verify SHA-256
`2f049e91c20f55b3967655c2828f4188cef4bc13108fd3a6d0407046375954b4`.

## Publication boundary

Only five payloads in this bundle currently have a documented public
redistribution basis: the MATH training parquet, MATH-500, AQUA, GSM8K, and
HumanEval+. The other eight payloads remain private-only because their data
license or complete upstream rights chain has not been established:
AIME-2025, AMC23, MAWPS, SVAMP, KodCode, MBPP+, LiveCodeBench, and
BigCodeBench.

The KodCode source manifest declares `CC-BY-NC-4.0`, but that declaration does
not establish a complete redistribution chain for all mixed upstream prompts
and tests. Evaluator source-code licenses likewise do not automatically license
their embedded benchmark data.
