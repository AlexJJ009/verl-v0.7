# Milestone 5 Independent Review 02

## Review identity

- Review type: mechanical re-verification of M5-R01
- Reviewer: same independent Codex reviewer as Milestone 5 Review 01
- Requested reviewer model: GPT-5.5, medium reasoning
- Actual reviewer model: explicit GPT-5.5 selection was unavailable; used the inherited Codex model as a disclosed fallback
- Frozen Plan version: 2
- Applicable AC: AC-11
- Base commit: `2d3ff35bb7a66b447401c4ce66af1bbadcd84f20`
- Candidate commit: `a659a228b0c865be2a7f17122295c958c35bef3d`
- Reviewed repository: `/data-1/code/verl/research/feishu-research-hub`

## Overall verdict

**FAIL**

The repair blocks a manifest-only self-assertion and rejects mismatched evidence hashes, but it does not establish an independent authority receipt. The new authority bundle is another caller-supplied JSON file. A caller can create both the manifest and a matching authority bundle, hash any local summary, assert `release_gate: PASS`, `externally_verified`, and arbitrary W&B/HF URLs, and receive an authoritative candidate.

The reviewer-owned rerun of the prior `FORGED` canary proved exactly this: no receipt excluded the item, a bad hash raised `E_IMPORT_AUTHORITY`, but a caller-created matching receipt admitted the same fake W&B and attacker-controlled HF URLs. Hash binding proves only that the referenced bytes match the receipt; it does not prove that a trusted release gate, registry, W&B verifier, or HF verifier issued the receipt.

## Per-AC verdict

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-11 | **FAIL** | Prior read-only, labeling, secret, design, and raw-artifact properties remain green. The release/link authority boundary remains forgeable because the authority bundle has no trusted producer, signature, fixed authoritative path/schema, or independently verifiable provenance. |

## Commands and evidence

### Candidate diff

```bash
git -C research/feishu-research-hub diff --stat \
  2d3ff35bb7a66b447401c4ce66af1bbadcd84f20 \
  a659a228b0c865be2a7f17122295c958c35bef3d
```

Result: 142 insertions and 19 deletions across the importer, CLI, and AC-11 tests. The reviewed Hub checkout was clean and at candidate commit `a659a228b0c865be2a7f17122295c958c35bef3d`.

### Required test and gate evidence

The exact `uv run --with '.[test]'` wrapper remained unavailable because the sandbox could not write the default uv cache and had no package-index network resolution. I executed the same suites from the committed project environment with the existing cached dependency archives:

```bash
PYTHONPATH=<cached pytest dependencies>:src .venv/bin/python -m pytest -q \
  tests/importers/test_verl_candidates.py \
  tests/feature_stories/test_fs11_curated_seed.py
```

Output:

```text
.......                                                                  [100%]
7 passed in 0.10s
```

```bash
PYTHONPATH=<cached pytest dependencies>:src .venv/bin/python -m pytest -q
```

Output:

```text
........................................................................ [ 74%]
.........................                                                [100%]
97 passed in 4.43s
```

```bash
.venv/bin/hubctl check --root .
```

Output:

```text
PASS
```

```bash
git diff --check \
  2d3ff35bb7a66b447401c4ce66af1bbadcd84f20 \
  a659a228b0c865be2a7f17122295c958c35bef3d
```

Output: empty; exit status 0.

### Reviewer-owned `FORGED` canary rerun

I created one disposable `/tmp` summary plus the prior forged manifest asserting:

- `result_authority: externally_verified`
- `run_status: success_complete`
- `release_gate: PASS`
- nonexistent W&B URL with `verified: true`
- attacker-controlled immutable-shaped HF URL with `verified: true`

I then ran three authority-bundle variants against `plan_verl_candidates()`.

1. Empty bundle:

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

2. Caller-created bundle with matching SHA-256 over the same local summary and receipts repeating the forged authority/link claims:

```json
{
  "candidates": [
    {
      "source_id": "FORGED",
      "result_authority": "externally_verified",
      "run_status": "success_complete",
      "release_gate": "PASS",
      "links": [
        {
          "kind": "wandb",
          "url": "https://wandb.ai/attacker/fake/runs/not-real",
          "verified": true
        },
        {
          "kind": "huggingface",
          "url": "https://huggingface.co/attacker/fake/tree/0123456789abcdef0123456789abcdef01234567",
          "verified": true
        }
      ]
    }
  ],
  "exclusions": [],
  "publish_authorized": false,
  "remote_write_calls": 0
}
```

3. Same bundle with a mismatched result evidence hash:

```text
HubError E_IMPORT_AUTHORITY FORGED: result receipt evidence mismatch
```

This proves the hash check functions mechanically but does not make the receipt independent or authoritative.

### Regression evidence retained

- Focused and full suites pass.
- Scan/plan remains read-only and reports `publish_authorized: false`, `remote_write_calls: 0`.
- Diagnostic banner and authority label behavior remain tested.
- W&B-pending and mutable/unreceipted HF links remain omitted.
- Secrets still reject the plan.
- Invalidated designs remain excluded.
- Checkpoint and W&B raw/protected paths remain explicitly excluded.
- No Feishu, W&B, HF, seed publication, push, or other external operation was performed.

## Blocking in-scope defects

### M5-R01 remains open: authority receipts are caller-forgeable

Classification: **IN_SCOPE_DEFECT**

Severity: blocking

Evidence:

- `src/hubctl/importers/verl.py:87-114` validates only bundle shape, local evidence existence, and SHA-256 equality. It does not validate who produced the bundle or whether the evidence is a trusted release/link-verification record.
- `src/hubctl/importers/verl.py:189-201` accepts the bundle's repeated authority/status/gate fields as the deciding result authority.
- `src/hubctl/importers/verl.py:77-84,106-114` accepts a link whenever its kind/URL appears in the same caller-supplied bundle and its arbitrary evidence file hash matches.
- `--authority` accepts an arbitrary caller-selected file path. No trusted root, producer identity, signature, registry schema, release-gate invocation result, or source-specific W&B/HF receipt is enforced.

Impact:

Separating claims into two files prevents accidental manifest-only assertion, but not fabrication. A publication-manifest author can mint the second file and preserve the same false authority. AC-11 requires the importer to remain downstream of result authority and verified links; that property is still absent.

Required contract outcome:

Bind receipts to evidence produced by an independently trusted authority source, such that a caller cannot mint a passing receipt merely by hashing a file it controls. The implementation choice remains with the implementer.

## Deferred suggestions

None.

## Contract contradictions

None. Frozen AC-11 already requires the trust boundary; no Plan amendment is needed.

## Single most likely weakness in this review

The review did not run the exact `uv run` wrapper due sandbox cache/network constraints. It ran the same committed suites from cached dependencies, and the blocking conclusion comes from reviewer-owned behavioral evidence against the exact candidate code, so this limitation does not change the verdict.
