# Milestone 5 Independent Review 01

## Review identity

- Review type: Milestone Review
- Reviewer: fresh Codex subagent, independent of the implementer
- Requested reviewer model: GPT-5.5, medium reasoning
- Actual reviewer model: unavailable to this agent as an explicit selectable identity; used the available inherited Codex model as a disclosed fallback
- Frozen Plan version: 2
- Applicable AC: AC-11
- Base commit: `efee75e560e44f9484ce2dc7a7e3d85a757f4fb8`
- Candidate commit: `2d3ff35bb7a66b447401c4ce66af1bbadcd84f20`
- Reviewed repository: `/data-1/code/verl/research/feishu-research-hub`

## Overall verdict

**FAIL**

AC-11 requires the importer to be downstream of existing result authority and to include only verified external links. The candidate instead accepts authority claims entirely from the same untrusted importer manifest: `release_gate: PASS`, `result_authority: externally_verified`, and per-link `verified: true` are sufficient to produce an authoritative candidate. No trusted release-gate receipt, registry record, W&B evidence, or Hugging Face verification evidence is consulted or bound to the candidate.

Reviewer-owned adversarial evidence produced a candidate containing a nonexistent W&B URL and an attacker-controlled Hugging Face URL merely by self-asserting those fields. This can overstate an incomplete/unreleased result and fabricate external verification, directly violating frozen AC-11. The otherwise-green tests only reproduce the same trusted-manifest assumption and do not exercise this canary.

## Per-AC verdict

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-11 | **FAIL** | Read-only behavior, diagnostic labeling, secret rejection, invalid-design exclusion, and raw/protected path exclusion pass. However, release authority and external-link verification can be self-asserted by the manifest, so the importer is not demonstrably downstream of authoritative sources and can overstate results. |

## Commands and evidence

### Candidate inspection

```bash
git -C research/feishu-research-hub diff --stat \
  efee75e560e44f9484ce2dc7a7e3d85a757f4fb8 \
  2d3ff35bb7a66b447401c4ce66af1bbadcd84f20

git -C research/feishu-research-hub diff --name-status \
  efee75e560e44f9484ce2dc7a7e3d85a757f4fb8 \
  2d3ff35bb7a66b447401c4ce66af1bbadcd84f20
```

Relevant result: 397 inserted lines across the importer, CLI entry point, and two test modules. The reviewed Hub worktree was clean and at candidate commit `2d3ff35bb7a66b447401c4ce66af1bbadcd84f20`.

### Required test and gate evidence

The prompt's exact `uv run --with '.[test]' ...` command could not resolve packages in this sandbox because DNS/network access was unavailable, and the default uv cache path was read-only. I therefore used the already-created project environment plus the existing uv package archives to execute the same pytest suite without downloading or altering dependencies:

```bash
PYTHONPATH=<cached pytest dependencies>:src .venv/bin/python -m pytest -q \
  tests/importers/test_verl_candidates.py \
  tests/feature_stories/test_fs11_curated_seed.py
```

Output:

```text
.....                                                                    [100%]
5 passed in 0.15s
```

```bash
PYTHONPATH=<cached pytest dependencies>:src .venv/bin/python -m pytest -q
```

Output:

```text
........................................................................ [ 75%]
.......................                                                  [100%]
95 passed in 4.43s
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
  efee75e560e44f9484ce2dc7a7e3d85a757f4fb8 \
  2d3ff35bb7a66b447401c4ce66af1bbadcd84f20
```

Output: empty; exit status 0.

### Reviewer-owned release-authority canary

I created a disposable `/tmp` repository fixture containing one summary and a manifest with:

```json
{
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
```

Command shape:

```bash
tmpdir=$(mktemp -d /tmp/m5-review.XXXXXX)
PYTHONPATH=src .venv/bin/python - "$tmpdir" <<'PY'
# write the summary and manifest above, then call plan_verl_candidates(...)
PY
```

The importer returned:

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

This is a deterministic failure of the authority boundary, independent of live W&B, Hugging Face, Feishu, or other external access.

### Read-only and artifact-safety evidence

- `plan_verl_candidates()` exposes no adapter or publisher and returns `publish_authorized: false` plus `remote_write_calls: 0`.
- FS-11 records the complete temporary fixture tree before and after CLI planning and proves no new Hub entry or other file is created.
- Candidate tests prove diagnostic labeling, W&B-pending omission, immutable-HF URL shape filtering, secret rejection, invalidated-design exclusion, and explicit checkpoint/W&B raw-path exclusions.
- The full test suite and `hubctl check` passed.

These are positive AC-11 properties, but they do not cure the blocking result-authority failure.

## Blocking in-scope defects

### M5-R01: Importer trusts self-asserted release and verification status

Classification: **IN_SCOPE_DEFECT**

Severity: blocking

Evidence:

- `src/hubctl/importers/verl.py:143-153` reads `result_authority`, `run_status`, and `release_gate` from each manifest item and treats literal `success_complete` plus `PASS` as sufficient release authority.
- `src/hubctl/importers/verl.py:62-79` treats manifest `verified: true` plus an HTTPS URL as sufficient W&B verification; for Hugging Face it adds only a commit-shaped URL regex.
- `src/hubctl/importers/verl.py:188-191` republishes those self-asserted authority fields into the candidate.
- `tests/importers/test_verl_candidates.py:27-79` supplies all positive authority and verification claims from the fixture manifest itself; it does not bind them to trusted independent evidence.

Impact:

Any manifest author can convert a failed, incomplete, or unreleased result into `authoritative_local`/`externally_verified`, or attach a fake W&B/HF link, without providing evidence from the existing release gate, result registry, W&B, or Hugging Face. Because a later explicit publish action can consume these candidates, `publish_authorized: false` at scan time does not make the curated candidate truthful.

Required contract outcome:

The importer must derive or validate release and link authority from trusted source evidence rather than accepting booleans/status strings from the publication manifest. Exact implementation is for the implementer, not this review.

## Deferred suggestions

None.

## Contract contradictions

None. AC-11 already requires the missing trust boundary; no Plan amendment is needed.

## Single most likely weakness in this review

The review did not execute the exact `uv run --with '.[test]'` wrapper because the sandbox could not use the read-only uv cache and had no network resolution. It ran the same tests from the committed project environment with cached dependencies, and the blocking verdict rests on a separate reviewer-owned deterministic canary, so this infrastructure limitation does not affect the finding.
