# Milestone 5 Independent Review 03

## Review identity

- Review type: post-convergence independent Milestone Review
- Reviewer: fresh Codex subagent, independent of the implementer
- Requested reviewer model: GPT-5.5, medium reasoning
- Actual reviewer model: GPT-5.5 was unavailable as a selectable model; used the available inherited Codex model as a disclosed fallback
- Frozen Plan version: 2
- Applicable AC: AC-11
- Base commit: `9e81e4ee9be33c7acc19c4bd88492f2d738f051e`
- Candidate commit: `6fca611d12caaec97c5efd4b72f950c6eec1a977`
- Reviewed repository: `/data-1/code/verl/research/feishu-research-hub`

## Overall verdict

**PASS**

Candidate `6fca611` closes the caller-forgeable authority-source defect from Reviews 01 and 02. `plan_verl_candidates` now accepts exactly `manifest_path` and `repo_root`; the `import-verl` CLI accepts no authority/source-path override. The release-gate script/state, W&B receipts, and HF manifest root are fixed repository-relative locations derived from `repo_root`.

The focused AC-11 suite passes 9 tests. Reviewer-owned CLI canaries reject both a generic `--authority` override and a concrete `--release-gate-script` override before manifest processing. Code inspection confirms the remaining fixed-source behavior: release-gate subprocess invocation uses the fixed script and fixed state; W&B admission requires an exact URL in the fixed receipt ledger plus an existing synced marker; HF admission requires uploaded and verified flags, a 40-character commit SHA embedded in the commit URL, immutable URL shape, and a manifest contained under the fixed HF root.

## Per-AC verdict

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-11 | **PASS** | The authority inputs are no longer caller-selectable through the Python or CLI surfaces; fixed-source release, W&B, and HF checks remain in place; all 9 focused importer/FS-11 tests pass; scan/plan remains read-only and preserves the prior curation properties. |

## Commands and evidence

### Candidate identity and diff inspection

```bash
git rev-parse HEAD
git show --stat --oneline 6fca611d12caaec97c5efd4b72f950c6eec1a977
git diff --check 9e81e4ee9be33c7acc19c4bd88492f2d738f051e 6fca611d12caaec97c5efd4b72f950c6eec1a977
git diff --unified=70 9e81e4ee9be33c7acc19c4bd88492f2d738f051e 6fca611d12caaec97c5efd4b72f950c6eec1a977 -- src tests/importers/test_verl_candidates.py tests/feature_stories/test_fs11_curated_seed.py
```

Relevant evidence:

```text
6fca611d12caaec97c5efd4b72f950c6eec1a977
6fca611 fix: pin project import authorities
4 files changed, 41 insertions(+), 37 deletions(-)
```

`git diff --check` produced no output and exited 0. The Hub worktree was clean.

### Focused AC-11 tests

The project's `.venv` does not contain pytest. I used the existing read-only uv package archives to run the committed tests without downloading packages or modifying the environment:

```bash
PYTHONPATH=<existing uv pytest dependency archives>:src \
  .venv/bin/python -m pytest -q \
  tests/importers/test_verl_candidates.py \
  tests/feature_stories/test_fs11_curated_seed.py
```

Output:

```text
.........                                                                [100%]
9 passed in 0.52s
```

The previously recorded 99-test full-suite and root-check PASS evidence was not reclassified as reviewer-owned evidence; this review used the focused suite and direct canaries requested for the converged defect.

### Python API and fixed-source inspection

```bash
.venv/bin/python - <<'PY'
import inspect
from hubctl.importers.verl import plan_verl_candidates, _PROJECT_AUTHORITY_PATHS
print(inspect.signature(plan_verl_candidates))
print({k: str(v) for k, v in _PROJECT_AUTHORITY_PATHS.items()})
PY
```

Output:

```text
(manifest_path: 'Path | str', repo_root: 'Path | str') -> 'dict[str, Any]'
{'release_gate_script': 'scripts/training_result_release_gate.py',
 'release_gate_state': '.hub-authority/training_release_gate.jsonl',
 'wandb_receipts': '.hub-authority/wandb_verified.jsonl',
 'hf_manifests_root': '.hub-authority/hf_manifests'}
```

Inspection of `src/hubctl/importers/verl.py` at the candidate confirms all four paths are resolved from `repo_root`. No function argument can replace them.

### Reviewer-owned CLI override canaries

```bash
.venv/bin/hubctl import-verl --manifest x --repo-root y --authority forged.json
.venv/bin/hubctl import-verl --manifest x --repo-root y --release-gate-script forged.py
```

Both returned exit status 2 before reading the nonexistent manifest:

```text
hubctl: error: unrecognized arguments: --authority forged.json
hubctl: error: unrecognized arguments: --release-gate-script forged.py
```

The parser defines only `--manifest`, `--repo-root`, and `--json` for `import-verl`; therefore the other former source overrides (`--release-gate-state`, `--wandb-receipts`, and `--hf-manifests-root`) are absent by construction as well.

### Fixed authority behavior

- Release authority loads `.hub-authority/training_release_gate.jsonl`, requires a matching latest `success_complete` event, and invokes `python3 scripts/training_result_release_gate.py --state .hub-authority/training_release_gate.jsonl check --run-name <name>` using paths resolved from the supplied project root.
- W&B authority loads only `.hub-authority/wandb_verified.jsonl`; a URL is admitted only when the receipt kind is `wandb_sync_verified`, the receipt URL matches exactly, and its synced marker currently exists as a file.
- HF authority resolves the manifest only beneath `.hub-authority/hf_manifests`, rejects traversal/missing files, and requires both `upload.uploaded` and `upload.verified`, a 40-character commit SHA embedded in `commit_url`, and the importer's immutable HF URL check.
- The prior AC-11 properties remain covered by the focused suite: manifest-only forged authority is excluded; unreceipted links are omitted; diagnostic labeling, secret rejection, invalid-design exclusion, raw/protected-path exclusion, immutable HF filtering, zero remote writes, and no filesystem mutation in the FS-11 plan path remain green.

## Blocking in-scope defects

None.

The prior M5-R01 defect is closed: a caller can choose which repository to inspect via the required `repo_root`, but cannot independently substitute authority files or scripts within that repository through the importer API or CLI. Treating an arbitrary untrusted directory as the authoritative project root is outside this importer's stated invocation contract; within the selected project, authority locations are fixed.

## Deferred suggestions

None.

## Contract contradictions

None.

## Single most likely weakness in this review

I did not rerun the full 99-test suite or `hubctl check`; the verdict is based on the candidate diff, reviewer-owned API/CLI canaries, and the 9 focused AC-11 tests. This is sufficient for the converged authority-source defect but provides less broad regression evidence than the prompt's complete command chain.
