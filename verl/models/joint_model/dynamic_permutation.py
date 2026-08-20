"""Target-preserving dynamic permutation for weak-model logits.

The transform is stateless.  For every active token row it hashes the explicit
training/update/sample/token identity into two values: a cyclic offset selecting
exactly ``floor(rho * (V - 1))`` non-target coordinates and a non-zero rotation
of those selected coordinates.  A rotation is an exact fixed-point-free cycle,
so the target logit and complete row value multiset are preserved.

Only ``O(row_chunk_size * floor(rho * (V - 1)))`` integer indices are
materialized.  The implementation never constructs a ``[token_rows, vocab]``
permutation tensor.  ``rho=0`` returns the input object before hashing or
allocation.  Because no process-global RNG is used, checkpoint/resume only
needs to restore the explicit identity fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_HASH_MODULUS = 2_147_483_647
_HASH_MULTIPLIER = 48_271


@dataclass(frozen=True)
class DynamicPermutationTelemetry:
    requested_rho: float
    realized_rho: float
    active_rows: int
    selected_per_row: int
    selected_coordinates: int
    fixed_points: int
    target_mismatches: int
    audited_rows: int
    max_entropy_error: float
    max_multiset_error: float
    invariant_failures: int

    def as_metrics(self) -> dict[str, float]:
        prefix = "dynperm/"
        return {
            f"{prefix}requested_rho": self.requested_rho,
            f"{prefix}realized_rho": self.realized_rho,
            f"{prefix}active_rows": float(self.active_rows),
            f"{prefix}selected_per_row": float(self.selected_per_row),
            f"{prefix}selected_coordinates": float(self.selected_coordinates),
            f"{prefix}fixed_points": float(self.fixed_points),
            f"{prefix}target_mismatches": float(self.target_mismatches),
            f"{prefix}audited_rows": float(self.audited_rows),
            f"{prefix}max_entropy_error": self.max_entropy_error,
            f"{prefix}max_multiset_error": self.max_multiset_error,
            f"{prefix}invariant_failures": float(self.invariant_failures),
        }


def validate_dynamic_permutation_rho(rho: float) -> float:
    """Return ``rho`` as a float or fail closed on an invalid dose."""

    if isinstance(rho, bool):
        raise ValueError("weak-logit permutation rho must be a finite numeric value in [0, 1]")
    try:
        value = float(rho)
    except (TypeError, ValueError) as exc:
        raise ValueError("weak-logit permutation rho must be a finite numeric value in [0, 1]") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"weak-logit permutation rho must be finite and in [0, 1], got {rho!r}")
    return value


def _mix_identity(
    sample_ids: torch.Tensor,
    token_positions: torch.Tensor,
    *,
    base_seed: int,
    global_step: int,
    actor_update_index: int,
) -> torch.Tensor:
    """Hash explicit row identity with arithmetic stable across CPU/GPU ranks."""

    mixed = torch.full_like(sample_ids, int(base_seed) % _HASH_MODULUS, dtype=torch.int64)
    for component in (
        int(global_step),
        int(actor_update_index),
        sample_ids.to(dtype=torch.int64),
        token_positions.to(dtype=torch.int64),
    ):
        if isinstance(component, torch.Tensor):
            component = torch.remainder(component, _HASH_MODULUS)
        else:
            component %= _HASH_MODULUS
        mixed = torch.remainder(mixed * _HASH_MULTIPLIER + component, _HASH_MODULUS)
    return mixed


def _empty_telemetry(rho: float, *, active_rows: int = 0) -> DynamicPermutationTelemetry:
    return DynamicPermutationTelemetry(
        requested_rho=rho,
        realized_rho=0.0,
        active_rows=active_rows,
        selected_per_row=0,
        selected_coordinates=0,
        fixed_points=0,
        target_mismatches=0,
        audited_rows=0,
        max_entropy_error=0.0,
        max_multiset_error=0.0,
        invariant_failures=0,
    )


def target_preserving_dynamic_permutation(
    weak_logits: torch.Tensor,
    target_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    token_positions: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    rho: float,
    base_seed: int,
    global_step: int,
    actor_update_index: int,
    row_chunk_size: int = 8,
    audit_rows: int = 0,
    entropy_atol: float = 2e-6,
    multiset_atol: float = 0.0,
) -> tuple[torch.Tensor, DynamicPermutationTelemetry]:
    """Permute selected non-target weak logits using a stateless keyed cycle.

    All identity tensors must match ``weak_logits.shape[:-1]``.  Only rows with
    ``active_mask=True`` are changed.  Runtime audits are deliberately bounded;
    exhaustive invariant checks belong in CPU tests.
    """

    rho = validate_dynamic_permutation_rho(rho)
    if weak_logits.ndim < 2:
        raise ValueError("weak_logits must have at least one row dimension and one vocabulary dimension")
    row_shape = weak_logits.shape[:-1]
    for name, value in (
        ("target_ids", target_ids),
        ("sample_ids", sample_ids),
        ("token_positions", token_positions),
        ("active_mask", active_mask),
    ):
        if tuple(value.shape) != tuple(row_shape):
            raise ValueError(f"{name} shape {tuple(value.shape)} must match logit rows {tuple(row_shape)}")
    if row_chunk_size <= 0:
        raise ValueError(f"row_chunk_size must be positive, got {row_chunk_size}")
    if audit_rows < 0:
        raise ValueError(f"audit_rows must be non-negative, got {audit_rows}")

    flat_active = active_mask.reshape(-1).to(dtype=torch.bool)
    active_count = int(flat_active.sum().item())
    # The no-op exits before hashing, permutation-index allocation, or any RNG use.
    if rho == 0.0 or active_count == 0:
        return weak_logits, _empty_telemetry(rho, active_rows=active_count)

    vocab_size = weak_logits.shape[-1]
    non_target_count = vocab_size - 1
    selected_count = math.floor(rho * non_target_count)
    if selected_count == 1:
        raise ValueError(
            f"rho={rho} selects exactly one of {non_target_count} non-target coordinates; "
            "a fixed-point-free permutation cannot be formed"
        )
    if selected_count < 2:
        return weak_logits, _empty_telemetry(rho, active_rows=active_count)

    flat_logits = weak_logits.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1).to(device=weak_logits.device, dtype=torch.int64)
    if bool(((flat_targets < 0) | (flat_targets >= vocab_size)).any().item()):
        raise ValueError("target_ids contains a vocabulary index outside weak_logits")
    flat_samples = sample_ids.reshape(-1).to(device=weak_logits.device, dtype=torch.int64)
    flat_positions = token_positions.reshape(-1).to(device=weak_logits.device, dtype=torch.int64)
    active_indices = flat_active.to(device=weak_logits.device).nonzero(as_tuple=False).squeeze(-1)
    active_targets = flat_targets.index_select(0, active_indices)
    row_keys = _mix_identity(
        flat_samples.index_select(0, active_indices),
        flat_positions.index_select(0, active_indices),
        base_seed=base_seed,
        global_step=global_step,
        actor_update_index=actor_update_index,
    )

    output = flat_logits.clone()
    offsets = torch.arange(selected_count, device=weak_logits.device, dtype=torch.int64)
    for start in range(0, active_count, row_chunk_size):
        stop = min(start + row_chunk_size, active_count)
        rows = active_indices[start:stop]
        keys = row_keys[start:stop]
        targets = active_targets[start:stop]
        compact_selected = torch.remainder(keys[:, None] + offsets[None, :], non_target_count)
        selected = compact_selected + (compact_selected >= targets[:, None]).to(torch.int64)
        # Keep consecutive optimizer updates observably dynamic for k > 2.
        # The row key still controls selection; this independent cycle key
        # advances by exactly one when either explicit update counter advances.
        cycle_keys = (
            flat_samples.index_select(0, rows) * 40_513
            + flat_positions.index_select(0, rows) * 97
            + int(base_seed)
            + int(global_step)
            + int(actor_update_index)
        )
        shifts = 1 + torch.remainder(cycle_keys, selected_count - 1)
        source_offsets = torch.remainder(offsets[None, :] + shifts[:, None], selected_count)
        source = selected.gather(1, source_offsets)
        source_values = flat_logits.index_select(0, rows).gather(1, source)
        transformed = flat_logits.index_select(0, rows).scatter(1, selected, source_values)
        output.index_copy_(0, rows, transformed)

    target_before = flat_logits.index_select(0, active_indices).gather(1, active_targets[:, None])
    target_after = output.index_select(0, active_indices).gather(1, active_targets[:, None])
    target_mismatches = int((target_before != target_after).sum().item())

    audited = min(int(audit_rows), active_count)
    max_entropy_error = 0.0
    max_multiset_error = 0.0
    if audited:
        # Lowest stateless keys give a deterministic, rank-independent bounded sample.
        audit_order = torch.argsort(row_keys)[:audited]
        audit_indices = active_indices.index_select(0, audit_order)
        audit_dtype = torch.float64 if weak_logits.dtype == torch.float64 else torch.float32
        before = flat_logits.index_select(0, audit_indices).to(audit_dtype)
        after = output.index_select(0, audit_indices).to(audit_dtype)
        # Canonicalize the reduction order before the entropy comparison.  The
        # transform is a permutation, so unsorted softmax reductions can differ
        # by a few float32 ulps solely because values are summed in a new order.
        before_sorted = torch.sort(before, dim=-1).values
        after_sorted = torch.sort(after, dim=-1).values
        before_entropy = torch.distributions.Categorical(logits=before_sorted).entropy()
        after_entropy = torch.distributions.Categorical(logits=after_sorted).entropy()
        max_entropy_error = float((before_entropy - after_entropy).abs().max().item())
        max_multiset_error = float((before_sorted - after_sorted).abs().max().item())

    invariant_failures = int(target_mismatches > 0)
    invariant_failures += int(max_entropy_error > entropy_atol)
    invariant_failures += int(max_multiset_error > multiset_atol)
    if invariant_failures:
        raise RuntimeError(
            "weak-logit Dynamic Permutation invariant audit failed: "
            f"target_mismatches={target_mismatches}, max_entropy_error={max_entropy_error}, "
            f"max_multiset_error={max_multiset_error}"
        )

    telemetry = DynamicPermutationTelemetry(
        requested_rho=rho,
        realized_rho=selected_count / non_target_count,
        active_rows=active_count,
        selected_per_row=selected_count,
        selected_coordinates=active_count * selected_count,
        fixed_points=0,
        target_mismatches=target_mismatches,
        audited_rows=audited,
        max_entropy_error=max_entropy_error,
        max_multiset_error=max_multiset_error,
        invariant_failures=0,
    )
    return output.reshape_as(weak_logits), telemetry
