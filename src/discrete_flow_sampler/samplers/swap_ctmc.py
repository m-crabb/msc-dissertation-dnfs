"""Swap-move CTMC: IS integrand ξ_t, one-event Euler step, sampler, c_t grid.

Mirrors ctmc.py / log_z_estimators.py for the swap move set. ξ_t and the Euler
step reuse the merged swap readout head; the residual/ξ_t read one i<j
representative per unordered pair so the single-pass reverse rate is exact.
"""

import functools

import torch
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.samplers._swap_neighbours import (
    SWAP_LOG_RATIO_CLAMP,
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.resampling import (
    ResamplingConfig,
    ResamplingStats,
    resample_if_needed,
)


def compute_xi_t_swap(x: Tensor, t: Tensor, head, target) -> Tensor:
    """Per-state swap IS integrand ξ_t (Eq. 8, swap form). (B,).

    E_{p_t}[ξ_t] = ∂_t log Z_t for any valid rate matrix (the outflow/inflow
    expectations cancel by a dummy-index relabel), so this is the swap c_t
    integrand and the eval IS-weight integrand.
    """
    pairs = upper_tri_pairs(x.shape[1], x.device)
    pair_scores = gather_pair_scores(head(x, t), pairs)  # (B, P), i<j
    return xi_t_swap_from_scores(pair_scores, x, t, target, pairs)


def xi_t_swap_from_scores(
    pair_scores: Tensor, x: Tensor, t: Tensor, target, pairs: Tensor
) -> Tensor:
    """ξ_t from already-gathered pair scores G[i,j], i<j. (B,).

    The head forward dominates eval wall-clock; the Euler step evaluates the
    head on the same (state, t) this integrand needs, so the eval loop passes
    the step's scores here instead of calling the head a second time.
    """
    G_plus = F.relu(pair_scores)
    neg_G_plus = F.relu(-pair_scores)
    log_ratio = target.swap_log_ratio(x, t, pairs).clamp(max=SWAP_LOG_RATIO_CLAMP)
    outflow = G_plus.sum(dim=-1)  # (B,)
    inflow = (neg_G_plus * log_ratio.exp()).sum(dim=-1)  # (B,)
    return target.dt_log_p_tilde_t(x, t) + outflow - inflow


def _euler_step_swap(head, state: Tensor, t_per_batch: Tensor, step_dt: Tensor,
                     stats: dict | None = None):
    """One-event swap Euler step. Returns (new_state, pair_scores) (B, n_pairs).

    Single global categorical over the i<j pairs plus a stay slot: at most one
    composition-preserving swap fires per step. Edges sharing a vertex conflict,
    so per-edge independent firing (single-site tau-leaping) is invalid here.
    Correct to O(dt^2). When Λ·dt > 1 the stay slot clamps to 0 and
    torch.multinomial renormalises the pair probabilities, so exactly one swap
    fires that step; logging the Λ·dt>1 clip fraction is the caller's
    responsibility (the train driver).

    The second return is the RAW gathered head output G[i,j] (relu applied
    internally where rates are needed), so the eval loop can reuse this one
    head call for the ξ_t integrand.

    `stats` (optional dict) accumulates the same transport counters as the
    matching step, so one-event trajectories get a measured jump budget too.
    This step has no thinning/rejection stage — the categorical draws the
    firing pair directly with probability rate·dt — so a fired event is both
    the proposal and the acceptance and "proposed" == "accepted" by
    construction (both keys are kept so downstream readers see one schema).
    "accepted_state_changing" excludes fired same-spin pairs: their swap is
    a state no-op, so counting them would overstate productive transport.
    """
    batch_size, d = state.shape
    pairs = upper_tri_pairs(d, state.device)  # (P, 2)
    n_pairs = pairs.shape[0]
    pair_scores = gather_pair_scores(head(state, t_per_batch), pairs)  # (B, P)
    forward_rates = F.relu(pair_scores)
    step_probs = (forward_rates * step_dt).clamp(0.0, 1.0)  # (B, P)
    stay_prob = (1.0 - step_probs.sum(dim=-1)).clamp(0.0, 1.0)  # (B,)
    categorical = torch.cat([step_probs, stay_prob[:, None]], dim=-1)  # (B, P+1)
    choice = torch.multinomial(categorical, num_samples=1).squeeze(-1)  # (B,)

    fired = choice < n_pairs
    chosen = pairs[choice.clamp(max=n_pairs - 1)]  # (B, 2); stay rows dummy
    site_i = torch.where(fired, chosen[:, 0], chosen.new_zeros(()))
    site_j = torch.where(fired, chosen[:, 1], chosen.new_zeros(()))
    if stats is not None:
        # Kept as device tensors (no per-step host sync), same style as the
        # matching step. Stay rows collapse to site 0 -> 0, whose "endpoint
        # spins" are trivially equal, but the fired mask excludes them anyway.
        fired_count = fired.sum()
        endpoint_spins_differ = (
            state.gather(1, site_i[:, None]) != state.gather(1, site_j[:, None])
        ).squeeze(1)
        stats["proposed"] = stats.get("proposed", 0) + fired_count
        stats["accepted"] = stats.get("accepted", 0) + fired_count
        stats["accepted_state_changing"] = (
            stats.get("accepted_state_changing", 0)
            + (fired & endpoint_spins_differ).sum()
        )
        stats["state_steps"] = stats.get("state_steps", 0) + batch_size
    # Branch-free swap-or-identity permutation per row: stay rows map site
    # 0 -> 0 (a no-op), so no `.any()`/`nonzero()` host-device sync.
    perm = torch.arange(d, device=state.device).expand(batch_size, d).clone()
    perm.scatter_(1, site_i[:, None], site_j[:, None])
    perm.scatter_(1, site_j[:, None], site_i[:, None])
    return state.gather(1, perm), pair_scores


def _vertex_disjoint_matching(proposed, priority, pairs, d, max_rounds=8):
    """Extract a vertex-disjoint subset (matching) of the `proposed` pairs.

    Luby-style rounds: a pair is a winner iff it holds the highest priority
    among all still-active pairs at BOTH its endpoints; winners are then
    vertex-disjoint by construction (distinct priorities ⇒ a vertex is the max
    for ≤1 pair). Winning vertices are retired and pairs touching them
    deactivated, then repeat. Driving toward a MAXIMAL matching drops only
    genuinely unresolvable conflicts, keeping the per-pair firing rate close to
    the proposal rate (the property the IS weight relies on). Returns a (B, P)
    bool mask ⊆ `proposed`.
    """
    batch_size, n_pairs = proposed.shape
    idx_i = pairs[:, 0].view(1, n_pairs).expand(batch_size, n_pairs)
    idx_j = pairs[:, 1].view(1, n_pairs).expand(batch_size, n_pairs)
    neg_inf = torch.finfo(priority.dtype).min
    active = proposed.clone()
    accepted = torch.zeros_like(proposed)
    for _ in range(max_rounds):
        if not active.any():
            break
        pr = torch.where(active, priority, neg_inf)  # (B, P)
        vmax = torch.full(
            (batch_size, d), neg_inf, dtype=priority.dtype, device=priority.device
        )
        vmax.scatter_reduce_(1, idx_i, pr, reduce="amax", include_self=True)
        vmax.scatter_reduce_(1, idx_j, pr, reduce="amax", include_self=True)
        win = active & (pr == vmax.gather(1, idx_i)) & (pr == vmax.gather(1, idx_j))
        accepted |= win
        used = torch.zeros(
            (batch_size, d), dtype=priority.dtype, device=priority.device
        )
        win_f = win.to(priority.dtype)
        used.scatter_reduce_(1, idx_i, win_f, reduce="amax", include_self=True)
        used.scatter_reduce_(1, idx_j, win_f, reduce="amax", include_self=True)
        endpoint_used = (used.gather(1, idx_i) > 0) | (used.gather(1, idx_j) > 0)
        active = active & ~win & ~endpoint_used
    return accepted


def _apply_swaps(state: Tensor, accepted: Tensor, pairs: Tensor) -> Tensor:
    """Apply a matching of swaps simultaneously via a site permutation.

    `accepted` is vertex-disjoint, so each (row, site) is reassigned at most
    once — the permutation has no collisions and every row stays on the slice.
    Non-accepted pairs scatter into a dummy slot d that is dropped before the
    gather, so no `nonzero()` host-device sync is needed.
    """
    batch_size, d = state.shape
    site_i = pairs[:, 0].unsqueeze(0).expand(batch_size, -1)
    site_j = pairs[:, 1].unsqueeze(0).expand(batch_size, -1)
    dummy = torch.full_like(site_i, d)
    perm = (
        torch.arange(d + 1, device=state.device).expand(batch_size, d + 1).clone()
    )
    perm.scatter_(
        1, torch.where(accepted, site_i, dummy), torch.where(accepted, site_j, dummy)
    )
    perm.scatter_(
        1, torch.where(accepted, site_j, dummy), torch.where(accepted, site_i, dummy)
    )
    return state.gather(1, perm[:, :d])


def _euler_step_swap_matching(head, state: Tensor, t_per_batch: Tensor, step_dt,
                              stats: dict | None = None):
    """Multi-event swap Euler step: fire a vertex-disjoint matching of pairs.

    Thin every pair by its firing probability rate·dt, then keep a random
    vertex-disjoint matching of the proposals. Returns (new_state,
    pair_scores) to match `_euler_step_swap`'s contract, so `sample_swap_ctmc`
    can swap the two step kinds. Correct to O(dt): proposal conflicts are O(dt²)
    as dt → 0, so this collapses to the one-event step in that limit. The caller
    controls dt to hold the expected events per site per step ≤ 0.1.

    `stats` (optional dict) accumulates the step's own fidelity numbers —
    proposed/accepted swap counts and states visited, kept as device tensors
    so no per-step host sync — because this is the RUNNING step's only
    faithfulness record: `lambda_dt_clipped_frac` in the training log gates
    the one-event step, which this function replaces, and the Luby matching
    silently drops proposals still contested after its round budget.
    Without a matching-native diagnostic a d256 run can sit outside its
    validated envelope unnoticed; the drop fraction this feeds is that
    certificate.
    """
    batch_size, d = state.shape
    pairs = upper_tri_pairs(d, state.device)
    pair_scores = gather_pair_scores(head(state, t_per_batch), pairs)  # (B, P)
    fire_prob = (F.relu(pair_scores) * step_dt).clamp(0.0, 1.0)
    proposed = torch.bernoulli(fire_prob).bool()
    priority = torch.rand(batch_size, pairs.shape[0], device=state.device)
    accepted = _vertex_disjoint_matching(proposed, priority, pairs, d)
    if stats is not None:
        # "accepted" includes same-spin pairs whose swap leaves the state
        # unchanged (Swap2(x,i,j) = x when x_i == x_j), so it overstates
        # productive transport; "accepted_state_changing" masks to pairs
        # whose endpoint spins differ at fire time — the honest jump count.
        endpoint_spins_differ = state[:, pairs[:, 0]] != state[:, pairs[:, 1]]
        stats["proposed"] = stats.get("proposed", 0) + proposed.sum()
        stats["accepted"] = stats.get("accepted", 0) + accepted.sum()
        stats["accepted_state_changing"] = (
            stats.get("accepted_state_changing", 0)
            + (accepted & endpoint_spins_differ).sum()
        )
        stats["state_steps"] = stats.get("state_steps", 0) + batch_size
    return _apply_swaps(state, accepted, pairs), pair_scores


def sample_swap_ctmc(
    head,
    x0: Tensor,
    ts: Tensor,
    *,
    return_log_weights: bool = False,
    return_all_states: bool = False,
    target=None,
    multi_event: bool = False,
    resampling: ResamplingConfig | None = None,
    matching_stats: dict | None = None,
    return_cv_integrand: bool = False,
):
    """Swap-CTMC trajectory sampler. Same contract as `ctmc.sample_ctmc`.

    Composition-preserving: every state stays on the fixed-N slice. IS weights
    accumulate ξ_t·dt at the left endpoint (Eq. 8, swap form).

    `multi_event=True` uses the vertex-disjoint-matching step (many swaps/step,
    O(d) trajectory length at scale); the default one-event step fires ≤1
    swap/step (O(d²) steps at the critical coupling).

    `matching_stats` (optional dict) accumulates the step's transport
    counters — proposed/accepted/state-changing swaps and states visited —
    for BOTH step kinds (see `_euler_step_swap_matching` and
    `_euler_step_swap` for each step's counting semantics). Feeds the
    `proposal_drop_frac` / `events_per_site_per_step` training-log columns
    and the eval-time jumps-per-site budget. None (the default) skips all
    accumulation — the historical behaviour, bit-for-bit.

    `resampling` (needs `target`, and one of `return_log_weights` /
    `return_all_states`) enables adaptive systematic resampling of the
    particle batch when interim ESS < τ·B (`samplers.resampling`).
    Resampling duplicates whole on-manifold rows, so composition stays
    bit-exact. Two modes:

      * with `return_log_weights=True` — the EVAL mode. Return becomes
        (x_final, log_weights, ResamplingStats); the final-segment
        log_weights feed `smc_log_z_estimate` together with the stats.
      * with `return_all_states=True` — the TRAINING-ROLLOUT mode (LEAPS
        Alg. 1 lines 11-14, whose trajectories Alg. 2 line 5 trains on).
        Weights are accumulated internally to drive the trigger ONLY, and
        the return is (trajectory, ResamplingStats) — deliberately WITHOUT
        the weights. They are a per-segment residue after the resets, so a
        caller that read them as the trajectory's IS weights would silently
        drop every banked increment; anything wanting a log Ẑ must use the
        eval mode. Each slice is recorded AFTER that step's checkpoint, so
        `trajectory[k]` is the equally-weighted ensemble that CONTINUES
        from t_k — the measure whose plain batch mean estimates E_{p_t}[·]
        (the `rollout_resample_ess_fraction` config field carries the full
        argument for why c_t needs exactly that). Slices already written
        are never rewritten with the ancestor permutation: replaying the
        genealogy backwards would replace each earlier slice's filtering
        marginal p_s with a smoothing one tilted by future weights, and
        nothing downstream consumes a trajectory as a path — the buffer
        stores (state, t) pairs and c_t is a per-slot mean, so only the
        per-slot marginal has to be right.

    `return_cv_integrand=True` requires `return_all_states=True`, `target`
    and resampling OFF: reuse is certified only for plain buffer rollouts.
    Return (trajectory, cv_integrand), adding (T, B) per-slot ξ_t (Eq. 8,
    swap form) via `xi_t_swap_from_scores` on the Euler step's pair scores;
    only the final slot needs a fresh head call. Bit-identical to sequential
    (chunk_rows=None) `compute_c_t_grid_swap`: same tensors/arithmetic, no
    extra RNG consumption (tests/test_cv_integrand_reuse.py). At d256 this
    removes 127 of 128 c_t-grid head forwards per outer cycle (~7-8 h eager
    per 16x16 CV run).
    """
    if return_log_weights and target is None:
        raise ValueError("sample_swap_ctmc(return_log_weights=True) requires `target`.")
    if return_log_weights and return_all_states:
        raise ValueError(
            "return_all_states and return_log_weights are mutually exclusive."
        )
    if resampling is not None and not (return_log_weights or return_all_states):
        raise ValueError(
            "sample_swap_ctmc(resampling=...) requires return_log_weights=True "
            "or return_all_states=True — the trigger lives on the weights, and "
            "there is nothing to hand back from a bare final state."
        )
    if resampling is not None and target is None:
        raise ValueError(
            "sample_swap_ctmc(resampling=...) requires `target`: the ESS "
            "trigger reads log-weights, which are accumulated from ξ_t."
        )
    if return_cv_integrand and not return_all_states:
        raise ValueError(
            "sample_swap_ctmc(return_cv_integrand=True) requires "
            "return_all_states=True: the integrand slots are the "
            "trajectory's grid slots."
        )
    if return_cv_integrand and target is None:
        raise ValueError(
            "sample_swap_ctmc(return_cv_integrand=True) requires `target`: "
            "ξ_t reads the annealing density."
        )
    if return_cv_integrand and resampling is not None:
        raise ValueError(
            "sample_swap_ctmc(return_cv_integrand=True) requires resampling "
            "OFF: the reuse is certified only for the plain buffer rollout."
        )

    state = x0.clone()
    batch_size, d = state.shape
    # The trigger needs weights even when the caller does not want them back.
    accumulate_log_weights = return_log_weights or resampling is not None
    log_weights = (
        torch.zeros(batch_size, dtype=state.dtype, device=state.device)
        if accumulate_log_weights
        else None
    )
    if return_all_states:
        trajectory = torch.empty(
            (len(ts), batch_size, d), dtype=state.dtype, device=state.device
        )
        trajectory[0] = state
    cv_integrand = (
        torch.empty(
            (len(ts), batch_size), dtype=state.dtype, device=state.device
        )
        if return_cv_integrand
        else None
    )

    smc_stats = (
        ResamplingStats(
            log_z_increment=torch.zeros((), dtype=x0.dtype, device=x0.device)
        )
        if resampling is not None
        else None
    )

    if multi_event:
        step_fn = functools.partial(
            _euler_step_swap_matching, stats=matching_stats
        )
    else:
        step_fn = functools.partial(_euler_step_swap, stats=matching_stats)
    pairs = upper_tri_pairs(d, x0.device)
    dts = ts[1:] - ts[:-1]
    for step in range(len(ts) - 1):
        step_dt = dts[step]
        t_per_batch = ts[step].expand(batch_size)
        new_state, step_pair_scores = step_fn(head, state, t_per_batch, step_dt)
        if accumulate_log_weights or return_cv_integrand:
            # ξ_t at the left endpoint reads the same head(state, t) the step
            # just computed; reusing its scores halves the head calls per step.
            xi_t = xi_t_swap_from_scores(
                step_pair_scores, state, t_per_batch, target, pairs
            )
            if accumulate_log_weights:
                log_weights = log_weights + xi_t * step_dt
            if return_cv_integrand:
                # Slot k of the CV grid IS this ξ_t: `state` here equals
                # trajectory[step] and t_per_batch equals t_grid[step].
                cv_integrand[step] = xi_t
        state = new_state
        # Checkpoint AFTER the state advance: the particle carrying log w(t+dt)
        # is x_{t+dt}, so that is the row set resampling duplicates/kills.
        if resampling is not None and step % resampling.check_every == 0:
            state, log_weights, log_z_increment, fired = resample_if_needed(
                state, log_weights, resampling.ess_threshold_fraction
            )
            if fired:
                smc_stats.log_z_increment = (
                    smc_stats.log_z_increment + log_z_increment
                )
                smc_stats.n_events += 1
                smc_stats.event_steps.append(step)
        # Recorded after the checkpoint so the slice is the ensemble that
        # continues (see the docstring). Unchanged when resampling is off:
        # `state` is returned untouched by a checkpoint that does not fire.
        if return_all_states:
            trajectory[step + 1] = state

    if return_cv_integrand:
        # The loop covered slots 0..T-2 (each step's scores are at the
        # slot it STARTED from); the final state never gets an Euler step,
        # so its slot is the one fresh head call of the whole grid.
        cv_integrand[-1] = compute_xi_t_swap(
            state, ts[-1].expand(batch_size), head, target
        )
        return trajectory, cv_integrand
    if return_all_states and resampling is not None:
        return trajectory, smc_stats
    if resampling is not None:
        return state, log_weights, smc_stats
    if return_log_weights:
        return state, log_weights
    if return_all_states:
        return trajectory
    return state



def n_slices(target) -> int:
    """Number of fixed-composition slices the target mixes over (1 without
    a registered grid, i.e. every specialist target)."""
    return len(getattr(target, "n_plus_values", ()) or ()) or 1


def slice_index_of(target, x: Tensor) -> Tensor:
    """Position of each row's slice in the target's composition grid. (B,)
    long; all zeros for a single-slice target.

    Swaps conserve n_plus, so a row's slice is readable off the state at
    any time, which is what lets the trainer look c_t up per row without
    storing a slice label in the replay buffer or the resume checkpoint.
    """
    counts = getattr(target, "n_plus_values", None)
    if not counts or len(counts) == 1:
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    n_plus = ((x + 1.0) * 0.5).sum(dim=-1).long()
    count_to_slice = torch.full((x.shape[1] + 1,), -1, dtype=torch.long,
                                device=x.device)
    count_to_slice[torch.tensor(counts, device=x.device)] = torch.arange(
        len(counts), device=x.device)
    slice_idx = count_to_slice[n_plus]
    if (slice_idx < 0).any():
        raise AssertionError(
            f"rows off the composition grid: n_plus in "
            f"{n_plus[slice_idx < 0][:5].tolist()}, grid {tuple(counts)}"
        )
    return slice_idx


def mean_per_slice(values: Tensor, slice_idx: Tensor, n_slices: int) -> Tensor:
    """Within-slice mean of a (T, M) integrand table over its M rows. (T, K).

    THE CORRECTION THIS ENCODES. On a slice mixture the
    residual for a row on slice C needs ∂_t log Z_t^{(C)}: swap dynamics
    hold every slice's mass fixed, so only each slice's conditional
    evolves; a single mixture-level ∂_t log Z_t cannot generally serve
    every slice's residual. E_{p_t^{(C)}}[ξ_t] = ∂_t log Z_t^{(C)} for any
    rates (Stein), so the estimator is a within-slice mean. The pooled
    mean over all rows (the earlier reduction) left every row an offset
    ∂_t log Z_t^{(C)} − mean_C ∂_t log Z_t^{(C)}, ~2 nats at d16 and ~18
    nats at d256 from the binomial base constant alone; with c_t detached
    that offset reaches the gradient as 2·Δ·E_q[∇ξ_t]. With exact ratios,
    E_{p_t^{(C)}}[∇ξ_t] = 0 by Stein cancellation; it need not vanish
    under the current model law or replayed past model laws.

    K == 1 is `values.mean(dim=-1)` bit-for-bit (the archived specialist
    path). A slice with no rows (~K·(1−1/K)^M, negligible at M ≥ 128)
    takes the pooled slot mean rather than NaN.
    """
    if n_slices == 1:
        return values.mean(dim=-1, keepdim=True)
    n_grid = values.shape[0]
    sums = torch.zeros(n_grid, n_slices, dtype=values.dtype, device=values.device)
    sums.index_add_(1, slice_idx, values)
    counts = torch.bincount(slice_idx, minlength=n_slices).to(values.dtype)
    means = sums / counts.clamp(min=1.0)
    pooled = values.mean(dim=-1, keepdim=True).expand(n_grid, n_slices)
    return torch.where(counts > 0, means, pooled)


def reduce_c_t_grid(integrand_per_t: Tensor, x_rows: Tensor, target) -> Tensor:
    """(T,) plain mean for a specialist, (T, K) within-slice means for a
    mixture. Shared by the estimator and the trainer's rollout-reuse path
    so both grids are the same reduction on the same values."""
    n_slice = n_slices(target)
    if n_slice == 1:
        return integrand_per_t.mean(dim=-1)
    return mean_per_slice(integrand_per_t, slice_index_of(target, x_rows), n_slice)


def compute_c_t_grid_swap(t_grid: Tensor, x_traj: Tensor, target, head, *,
                          mode, chunk_rows: int | None = None):
    """Per-time-slot c_t for the swap loss (mirror of compute_c_t_grid).

    mode='naive_mc'        -> c_t = mean_m ∂_t log p̃_t(x_t^{(m)})
    mode='control_variate' -> c_t = mean_m ξ_t^swap(x_t^{(m)})   (Eq. 8, swap form)

    Returns (c_t_grid, integrand_per_t). The grid is (T,) for a single-
    slice target — the archived contract, bit-identical — and (T, K) for a
    K-slice composition mixture, the mean taken WITHIN each slice (see
    `mean_per_slice` for why the pooled mean was wrong). Rows keep their
    slice for the whole trajectory, so `x_traj[0]` labels every slot.

    chunk_rows: None runs the per-slot
    sequential loop — n_grid integrand calls at outer_batch rows each, the
    byte-identical archived behaviour. When set, the (n_grid × outer_batch)
    integrand evaluations are flattened and computed in row-chunks of at
    most chunk_rows, cutting the per-call launch overhead that dominates
    the no-grad c_t phase at d256 (~75% of wall there is trajectory+c_t).
    The quantities are the SAME fp32 ops modulo batch-dim blocking, so
    parity vs the sequential path is pinned at the established 1e-5
    batch-blocking class (tests/test_c_t_grid_chunk.py; no quality change
    is permitted). The cap exists so
    the flattened batch stays inside GPU memory: at d256-MA a no-grad call
    peaks ~40 MB/row, so 512-2048 rows is the in-cap class on an 80 GB
    a100. Stateless and RNG-free, so no resume contract beyond wiring.
    """
    if mode not in ("naive_mc", "control_variate"):
        raise ValueError(f"Unknown mode {mode!r}")
    n_grid, outer_batch, _ = x_traj.shape
    integrand_per_t = torch.empty(
        (n_grid, outer_batch), dtype=x_traj.dtype, device=x_traj.device
    )
    if chunk_rows is not None:
        if isinstance(chunk_rows, bool) or int(chunk_rows) != chunk_rows:
            raise TypeError(
                f"chunk_rows must be an int or None, got {chunk_rows!r}"
            )
        chunk_rows = int(chunk_rows)
        if chunk_rows < 1:
            raise ValueError(
                f"chunk_rows must be >= 1 when set, got {chunk_rows}"
            )
        # Row (k, m) of the flattened layout is grid slot k, rollout row m:
        # reshape is row-major over (n_grid, outer_batch), so the matching
        # time is t_grid[k] repeated outer_batch times.
        n_rows = n_grid * outer_batch
        x_flat = x_traj.reshape(n_rows, -1)
        t_flat = t_grid.repeat_interleave(outer_batch)
        integrand_flat = integrand_per_t.view(n_rows)
        with torch.no_grad():
            for start in range(0, n_rows, chunk_rows):
                stop = min(start + chunk_rows, n_rows)
                if mode == "naive_mc":
                    integrand_flat[start:stop] = target.dt_log_p_tilde_t(
                        x_flat[start:stop], t_flat[start:stop]
                    )
                else:
                    integrand_flat[start:stop] = compute_xi_t_swap(
                        x_flat[start:stop], t_flat[start:stop], head, target
                    )
        return reduce_c_t_grid(integrand_per_t, x_traj[0], target), integrand_per_t
    with torch.no_grad():
        for k in range(n_grid):
            x_k = x_traj[k]
            t_k = t_grid[k].expand(outer_batch)
            if mode == "naive_mc":
                integrand_per_t[k] = target.dt_log_p_tilde_t(x_k, t_k)
            else:
                integrand_per_t[k] = compute_xi_t_swap(x_k, t_k, head, target)
    return reduce_c_t_grid(integrand_per_t, x_traj[0], target), integrand_per_t
