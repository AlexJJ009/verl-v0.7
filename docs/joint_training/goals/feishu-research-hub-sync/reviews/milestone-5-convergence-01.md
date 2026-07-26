# Milestone 5 Convergence Review 01

## Review identity

- Review type: Convergence Review
- Reviewer: fresh Codex subagent, independent of the implementer and prior reviewers
- Requested reviewer model: GPT-5.5, medium reasoning
- Actual reviewer model: GPT-5.5 was not available as a selectable model; used the available inherited Codex model as a disclosed fallback
- Frozen Plan version: 2
- Applicable AC: AC-11
- Base commit: `2d3ff35bb7a66b447401c4ce66af1bbadcd84f20`
- Candidate commit: `9e81e4ee9be33c7acc19c4bd88492f2d738f051e`
- Reviewed repository: `/data-1/code/verl/research/feishu-research-hub`

## Overall verdict

**FAIL — NOT CONVERGED**

Candidate `9e81e4e` removes the single arbitrary authority-bundle argument and blocks a manifest-only self-assertion. It does not close the caller-forgeable authority boundary. The CLI and Python API still accept caller-selected release-gate state, W&B receipt file, and HF manifest root. Even when the actual project `scripts/training_result_release_gate.py` is used, a caller-minted JSONL event passes that gate because the importer supplies the caller-selected state with `--state`. A caller can also mint a W&B receipt pointing at any existing file and an HF manifest containing self-asserted `uploaded=true`, `verified=true`, `commit_sha`, and `commit_url`.

Reviewer-owned evidence admitted `FORGED` with both nonexistent/attacker-controlled W&B and HF URLs. Therefore AC-11 remains open and a third ordinary fix round is not yet justified as a simple mechanical retry.

The convergence cause is **duplicated/caller-selected authority**: importer inputs still recreate release/link authority beside the project's existing authority sources. This is an in-scope architectural defect in the AC-11 trust boundary. It is not experiment-specific policy in a shared layer, reviewer scope expansion, a second independently useful outcome, or a contract contradiction.

## Per-AC verdict

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-11 | **FAIL** | Read-only planning, diagnostic labeling, failed/incomplete fail-closed behavior, secret rejection, invalid-design exclusion, raw/protected-path exclusion, and ordinary tests pass. The source-validation boundary remains caller-forgeable, so non-diagnostic results and external links are not reliably downstream of existing project authority. |

## Commands and evidence

### Candidate and contract inspection

```bash
git -C research/feishu-research-hub diff --stat \
  2d3ff35bb7a66b447401c4ce66af1bbadcd84f20 \
  9e81e4ee9be33c7acc19c4bd88492f2d738f051e
git -C research/feishu-research-hub diff \
  2d3ff35bb7a66b447401c4ce66af1bbadcd84f20 \
  9e81e4ee9be33c7acc19c4bd88492f2d738f051e -- \
  src/hubctl/importers/verl.py src/hubctl/__main__.py \
  tests/importers/test_verl_candidates.py \
  tests/feature_stories/test_fs11_curated_seed.py
```

Relevant result: 253 insertions and 19 deletions across four files. The API accepts `release_gate_script`, `release_gate_state`, `wandb_receipts`, and `hf_manifests_root` as inputs; the CLI exposes all four as caller-provided required arguments.

### Required tests and deterministic gate

The prompt's exact `uv run --with '.[test]' ...` chain could not resolve in this sandbox. The default uv cache was read-only, and a writable `/tmp` cache then failed on restricted DNS while fetching PyYAML. I ran the same committed test suites with the existing project environment and cached pytest dependencies:

```bash
PYTHONPATH=<cached pytest dependencies>:src .venv/bin/python -m pytest -q \
  tests/importers/test_verl_candidates.py \
  tests/feature_stories/test_fs11_curated_seed.py
```

Output:

```text
........                                                                 [100%]
8 passed in 0.51s
```

```bash
PYTHONPATH=<cached pytest dependencies>:src .venv/bin/python -m pytest -q
```

Output:

```text
........................................................................ [ 73%]
..........................                                               [100%]
98 passed in 4.81s
```

```bash
.venv/bin/hubctl check --root .
git diff --check \
  2d3ff35bb7a66b447401c4ce66af1bbadcd84f20 \
  9e81e4ee9be33c7acc19c4bd88492f2d738f051e
```

Output: `PASS`; `git diff --check` returned no output and exit status 0.

### Previous forged canary 1: manifest-only self-assertion

A disposable `/tmp` manifest asserted `externally_verified`, `success_complete`, `release_gate=PASS`, and fake W&B/HF URLs, but supplied an empty release-gate state and no verification evidence.

Result:

```json
{
  "candidates": [],
  "exclusions": [
    {
      "item_id": "FORGED",
      "reason": "missing_or_mismatched_authority_receipt",
      "target": "item"
    }
  ],
  "publish_authorized": false,
  "remote_write_calls": 0
}
```

This prior canary is now blocked.

### Previous forged canary 2: caller-minted authority evidence

The reviewer then supplied:

- the real project script `/data-1/code/verl/scripts/training_result_release_gate.py`;
- a caller-created JSONL state containing one `success_complete` event for `FORGED-RUN`;
- a caller-created W&B receipt whose `synced_marker` pointed to an arbitrary existing file;
- a caller-created HF manifest root containing `uploaded=true`, `verified=true`, and an attacker-controlled immutable-shaped commit URL.

The exact candidate API returned:

```json
{
  "candidate_ids": ["FORGED"],
  "links": [
    {
      "kind": "wandb",
      "url": "https://wandb.ai/attacker/fake/runs/no",
      "verified": true
    },
    {
      "kind": "huggingface",
      "url": "https://huggingface.co/attacker/fake/commit/0123456789abcdef0123456789abcdef01234567",
      "verified": true
    }
  ]
}
```

The project gate itself returned success because the importer invoked it with the caller-selected `--state`. This reproduces the prior caller-forgeable receipt failure through the new source arguments.

### Diagnostic and non-diagnostic behavior

Focused and full suites prove:

- diagnostic failed results remain candidates only with a visible banner beginning `Diagnostic only` and retain `result_authority=diagnostic`;
- non-diagnostic `failed` or `incomplete` results are excluded as `release_authority_blocked`;
- planning remains read-only with `publish_authorized=false` and `remote_write_calls=0`;
- secrets, invalidated designs, raw checkpoint/model/W&B artifacts, unverified W&B URLs, and mutable/unverified HF URLs remain rejected or omitted.

These properties pass but do not cure the source-authority defect.

## Blocking in-scope defects

### M5-R01 remains open: authority source selection is caller-forgeable

Classification: **IN_SCOPE_DEFECT**

Blocking invariant:

> A non-diagnostic candidate may be emitted only when its exact run and external links are derived from project-configured authority sources that the candidate manifest and importer caller cannot replace. Passing a canonical checker against caller-selected state is not sufficient. W&B evidence must bind the exact URL/run to a verification artifact produced by the project release workflow, and HF evidence must come from a project-configured manifest root whose producer and schema are authoritative; arbitrary CLI/API paths must not substitute for those sources.

Evidence:

- `src/hubctl/__main__.py:95-100,131-139` accepts every purported authority source as a caller-supplied path.
- `src/hubctl/importers/verl.py:192-206` accepts those paths without binding them to project configuration or validating provenance.
- `src/hubctl/importers/verl.py:242-258` invokes the checker with the caller-selected state.
- `src/hubctl/importers/verl.py:114-128` treats any receipt URL as verified when `synced_marker` names any existing file; it does not bind marker, run, URL, or trusted producer.
- `src/hubctl/importers/verl.py:134-152,259-264` trusts self-asserted HF booleans and commit fields from a manifest under a caller-selected root.

This is the same trust-boundary finding as the two prior rounds, not a new AC.

## Deferred suggestions

None.

## Contract contradictions

None. Frozen AC-11 already requires the importer to be downstream of existing result authority and verified links. No Plan amendment or split Goal is required to state the missing invariant.

## Single most likely weakness in this review

The review could not execute the exact `uv run --with '.[test]'` wrapper because the sandbox had a read-only default uv cache and restricted DNS. It ran the same committed suites from the existing environment and cached dependencies. More importantly, the blocking verdict rests on a reviewer-owned behavioral canary using the actual project release-gate script, so the wrapper limitation does not affect the trust-boundary result.
