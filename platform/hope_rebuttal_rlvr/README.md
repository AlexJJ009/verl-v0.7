# Hope entry for rebuttal MATH RLVR

This directory is the Layer-1/Layer-2 AFO entry for the ordinary-SFT versus
paper offline-WDL-SFT comparison. `submit_manifest.py` renders one isolated
`run.hope` + `jupyter.sh` stage per job and invokes `hope run run.hope` only
when an approved Hope-semantics receipt is supplied. Formal/pilot submission
also requires a detached SSH-Ed25519-signed G3 attestation from a reviewer key
allowlisted in the clean checkout. A formal H20 profile separately requires a
G4 signature from the same independently governed allowlist, binding both arm
terminal receipts and the actual staged resource projection. The checked-in
allowlist is currently empty, so real submission intentionally fails closed.

All Meituan workers are network-isolated. The launch path pins
`WANDB_MODE=offline`; it stores W&B data under the persistent `STATE_ROOT`, never
calls `wandb sync`, and records a SHA-256 manifest so the colleague can export
the offline run together with logs/checkpoints for later joint analysis.

The manifest keeps `root` as the storage-security boundary and requires three
controlled roots below it: `dataset_root`, `model_root`, and `state_root`.
Workers receive these as `DATASET_ROOT`, `MODEL_ROOT`, and `STATE_ROOT`.
Repository checkout remains at `$ROOT/$REPO_SUBPATH`; train/Math-7 inputs are
under `$DATASET_ROOT/data`, formal initialization is below `$MODEL_ROOT`, and
all persistent output/checkpoint/eval/log/offline-W&B/receipt/cache paths derive
from `$STATE_ROOT/verl-exp`. Pod-local Ray/vLLM/ZMQ/TMP paths remain under
`/tmp/rebuttal_rlvr`. The local registry DB and release-gate state are fixed
below `$STATE_ROOT/experiment_registry`; per-attempt release log/status and file
metrics are rebound below the receipt/log attempt directories, so inherited
host paths cannot escape the storage boundary.

For a colleague already inside an allocated eight-H20 worker, the minimal
direct entry is `bash run_colleague.sh R01|R02 20260727`. It uses the explicit
user-approved external-provenance assumption and does not require the signed
Hope submission bundle. It is a conditional checkpoint comparison, while the
manifest route below remains the audited parallel-queue route.

The colleague-facing entry is:

```bash
cp platform/hope_rebuttal_rlvr/handoff.env.example \
   platform/hope_rebuttal_rlvr/handoff.env
# Fill ROOT, DATASET_ROOT, MODEL_ROOT, STATE_ROOT, HANDOFF_BUNDLE_ROOT,
# and R01_MODEL_PATH after R01/G3/G4 arrive.
bash platform/hope_rebuttal_rlvr/run_handoff.sh
```

`run_handoff.sh` validates the pre-registered slots and expands the full
submitter invocation. The colleague does not manually assemble receipt flags.
The reviewed bundle uses fixed filenames below `HANDOFF_BUNDLE_ROOT`, so the
normal handoff uses one storage boundary plus three controlled roots and the
bundle/model slots, rather than one flag per receipt.
The machine-readable source of the known/pending split is
`handoff_registry.json`; unknown provenance remains explicitly pending rather
than being converted into a fake passing receipt.
Training completion then invokes the checked-in release hook automatically;
no separate manual publication command is part of the handoff.
The default operation is render-only:

```bash
python3 platform/hope_rebuttal_rlvr/submit_manifest.py \
  --manifest /absolute/path/batch.json \
  --render-only \
  --render-output /absolute/scratch/rendered
```

After G3 is independently approved, the formal/pilot invocation is:

```bash
python3 platform/hope_rebuttal_rlvr/submit_manifest.py \
  --manifest /absolute/path/approved-batch.json \
  --submit \
  --render-output /absolute/path/attempt-archive \
  --hope-semantics-receipt /absolute/path/hope-semantics.json \
  --g3-admission-receipt /absolute/path/g3-admission.json \
  --submission-ledger /absolute/path/global-submission-ledger.jsonl
```

Formal manifests require 18 unique cells: two arms, three initialization
pairs, and RL seeds `20260727/28/29`. A one-pair pilot requires six. Smoke
manifests may use Base only when every affected row says
`run_mode=smoke` and `allow_base_placeholder=true`; formal/pilot validation
rejects that combination.

Real submission remains blocked until the colleague records the installed Hope
CLI behavior in a reviewed semantics receipt: the run verb, scheduler-return
behavior, job-ID extraction, per-job status command, and a global all-user
active-job listing. Status must be exactly one recognized stdout line with
empty stderr. Listing headers/empty markers are literal, mutually exclusive
lines and every job row is parsed by full-line match with no ignored output.
Completion-blocking `hope run` semantics fail closed. The signed semantics
freeze one absolute global ledger, reconcile both old ledger jobs and unrelated
platform-active jobs, cap active jobs at 10, and default to 8.

For formal/pilot submission, G3 must additionally bind a successful terminal
path/image smoke receipt, status mapping, worker environment precedence,
client interruption reconciliation, review-evidence bytes, and the exact
semantics-receipt file hash. The reviewer key owner must differ from the live
OS submitter identity. Every attempt archives the approved manifest, bound
receipts/evidence, pre-submit-verified INI/shim bytes, stdout/stderr/return code,
and resulting AFO job ID.
