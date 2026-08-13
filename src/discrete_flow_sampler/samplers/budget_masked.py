"""Budget-masked masked-diffusion sampler on the fixed-composition fibre.

The masked-diffusion analogue of the swap CTMC: instead of restricting the
MOVE SET of a flip sampler to composition-preserving swaps, the reference
process reveals masked sites one at a time and restricts the SPECIES DRAW
to the remaining budget — reveal a uniformly chosen masked site, assign
species +1 with probability b/m (b = remaining +1 budget, m = masked
sites). Its terminal law is uniform on the fibre and every trajectory
carries the same reference constant N_+!(d-N_+)!/d! (the trajectory-
constant lemma, exhaustively verified in tests/test_budget_masked_reference
.py), which is what makes importance weights implementable: all reference
terms cancel under self-normalisation.

Three verified facts carried over from the derivation notes
(docs/reviews/2026-08-13-mdns-budget-preconditioner.md and
2026-08-13-constrained-wdce-derivation.md — git-excluded docs; the
maths lives in the tests named below):

1. Constrained Lemma 3: the optimally controlled generator unmasks site i
   to species s at rate gamma(t) * Pr_pi(X^i = s | X^UM) — the CONSTRAINED
   target's masked conditional. The reference's urn factor b/m cancels the
   value function's completion-count ratio C(m,b)/C(m-1,b-1) = m/b
   identically, so the network learns that conditional directly
   (tests/test_budget_preconditioner.py::
   test_value_function_tilt_equals_exact_masked_conditional).
2. The budget-tilted preconditioner V0 approximates it in closed form:
   logit(+1) - logit(-1) = log(b/(m-b)) + 4*sigma*f_i, with masked
   neighbours imputed at the urn mean (2b-m)/m. Exact at sigma = 0, at
   m = 1, and at exhausted budgets (same test module, 8 tests).
3. The WDCE loss transfers with no structural change: corruption is the
   unconstrained mask-independently kernel (order-assignment independence),
   and the population minimiser at every context is the Lemma-3 conditional
   (tests/test_budget_wdce.py, 8 tests).

Conventions: masked states are float tensors with +1.0/-1.0 spins and 0.0
at masked sites (0 doubles as the zero-imputation value, so the field
computation is a single matmul); adjacency is IsingTarget's symmetric 0/1
matrix (each undirected edge in both directions), giving the single-site
tilt E(+1) - E(-1) = 4*(A x)_i on the x^T A x energy.
"""
import torch
from torch import Tensor, nn
from torch.nn.functional import logsigmoid, softplus

# Pseudo-infinite logit for the boundary-budget branch of the budget-tilted
# preconditioner. log(b/(m-b)) diverges at b = 0 or b = m; the closed form
# and the exact delta agree in the limit, so the branch is float safety,
# not semantics (mirrors the pure-python reference). 30 saturates a float32
# sigmoid (~1e-13 from the boundary) while leaving room for a trunk logit
# on top without overflow.
BOUNDARY_LOGIT = 30.0


def masked_count_and_budget(
    x_masked: Tensor, n_plus_target: int
) -> tuple[Tensor, Tensor]:
    """Per-row (m, b): masked-site count and remaining +1 budget.

    Both are pure functions of the state — no bookkeeping to fall out of
    sync with the actual reveals (the failure mode of carrying budget as
    training-loop state; the corruption step of the WDCE loss masks
    arbitrary subsets, where incremental bookkeeping has no meaning).
    """
    masked_count = (x_masked == 0.0).sum(dim=1)
    budget = n_plus_target - (x_masked == 1.0).sum(dim=1)
    return masked_count, budget


def masked_state_features(x_masked: Tensor) -> Tensor:
    """Three-channel one-hot encoding (+1 / -1 / masked), shape (B, 3d).

    The trunk sees only this: b and m are computable from it, so a capable
    network CAN learn the budget structure — which is exactly what arm B of
    the gate measures (how much of it must be learned when the
    preconditioner does not supply it).
    """
    return torch.cat(
        [x_masked == 1.0, x_masked == -1.0, x_masked == 0.0], dim=1
    ).float()


class MaskedConditionalNet(nn.Module):
    """Minimal trunk Phi_theta for the masked conditional, MDNS-style.

    Outputs a per-site logit DIFFERENCE (logit(+1) - logit(-1)); the
    sampler's conditional is sigmoid(Phi_theta(x) + P(x)) with P the arm's
    preconditioner, exactly the paper's softmax(network + preconditioner)
    reduced to two species. The final layer is ZERO-INITIALISED so that at
    step 0 the sampler IS the preconditioner's law — the from-scratch
    mechanism the Fig.-10 ablation measures (with a random init the arms
    would differ by init noise as well as by preconditioner, confounding
    the comparison the gate exists to make).
    """

    def __init__(self, n_sites: int, hidden_width: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(3 * n_sites, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, n_sites),
        )
        final_layer = self.trunk[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, x_masked: Tensor) -> Tensor:
        return self.trunk(masked_state_features(x_masked))


def preconditioner_logit_diff(
    x_masked: Tensor,
    adjacency: Tensor,
    sigma: float,
    n_plus_target: int,
    mode: str,
) -> Tensor:
    """The arm-defining logit offset P(x), shape (B, d); rows are valid at
    masked sites only (unmasked positions are revealed, never drawn).

    mode = "budget_tilted": the derived V0,
        log(b/(m-b)) + 4*sigma*f_i,  f_i = (A x_imputed)_i,
    with masked neighbours imputed at the urn mean (2b-m)/m — the only
    imputation that is unbiased under the urn's exchangeable marginal.
    Boundary budgets take the pseudo-infinite branch (exact deltas).

    mode = "unconstrained": the paper's App. D.4 form verbatim —
    4*sigma*f_i with masked neighbours at zero and NO budget term. Blind to
    the budget by construction; the gate's arm C measures what that costs.

    mode = "none": zero (arm B, the Fig.-10 no-preconditioning mirror).

    Rejected alternative (documented in the derivation note): V1 imputes
    the other masked sites' urn mean CONDITIONED on the hypothesis at i;
    ~10% lower mean error, but needs a hypothesis-dependent field pass and
    shares every exactness limit with V0 — not worth the complication at
    gate scale.
    """
    if mode == "none":
        return torch.zeros_like(x_masked)
    if mode == "unconstrained":
        # masked sites are 0.0, so one matmul zero-imputes for free
        return 4.0 * sigma * (x_masked @ adjacency)
    if mode != "budget_tilted":
        raise ValueError(f"unknown preconditioner mode: {mode}")
    budget_logit, energy_term, interior = budget_tilt_components(
        x_masked, adjacency, sigma, n_plus_target
    )
    logit = budget_logit[:, None] + energy_term
    # boundary rows: the delta must not be diluted by the energy field
    boundary = ~interior
    logit[boundary] = budget_logit[boundary][:, None].expand(
        -1, x_masked.shape[1]
    )
    return logit


def budget_tilt_components(
    x_masked: Tensor, adjacency: Tensor, sigma: float, n_plus_target: int
) -> tuple[Tensor, Tensor, Tensor]:
    """V0's two logit terms, separately: (budget_logit (B,), energy_term
    (B, d), interior (B,) bool).

    budget_logit is log(b/(m-b)) on interior rows and the pseudo-infinite
    delta on boundary rows; energy_term is 4*sigma*f with urn-mean
    imputation. Split out so the gated variant (`GatedBudgetTiltOffset`)
    can scale the two mechanisms independently without duplicating the
    computation the ungated preconditioner is exhaustively tested on.
    """
    masked_count, budget = masked_count_and_budget(x_masked, n_plus_target)
    interior = (budget > 0) & (budget < masked_count)
    # urn-mean imputation: field = A x  +  urn_mean * (A masked_indicator)
    urn_mean = torch.where(
        interior,
        (2.0 * budget - masked_count) / masked_count.clamp(min=1),
        torch.zeros_like(budget, dtype=x_masked.dtype),
    )
    masked_indicator = (x_masked == 0.0).float()
    field = x_masked @ adjacency \
        + urn_mean[:, None] * (masked_indicator @ adjacency)
    budget_logit = torch.where(
        interior,
        torch.log(budget.clamp(min=1).float())
        - torch.log((masked_count - budget).clamp(min=1).float()),
        torch.where(
            budget <= 0,
            torch.full_like(budget, -BOUNDARY_LOGIT, dtype=x_masked.dtype),
            torch.full_like(budget, +BOUNDARY_LOGIT, dtype=x_masked.dtype),
        ),
    )
    return budget_logit, 4.0 * sigma * field, interior


class GatedBudgetTiltOffset(nn.Module):
    """V0 with learnable scales on its two mechanisms (the Amendment-01
    forensics' arm A': "keep the good init without the unlearning bill").

        P(x) = gate_budget * log(b/(m-b)) + gate_field * energy_term
        (interior rows; boundary rows keep the UNGATED exact deltas)

    Both gates initialise at 1.0, so step 0 is EXACTLY the V0 law — the
    first-pass finding was that V0's strong-but-imperfect near-boundary
    opinions must be partially cancelled by the trunk (arm C ends better
    than arm A precisely there); two scalars turn that cancellation into
    a 2-parameter descent instead of distributed trunk weights. The
    boundary branch stays ungated because it is exact at any coupling —
    a gate there could only unlearn a true delta. Rejected alternative:
    a context-conditioned gate g(m, b) (small MLP) — more capacity, but
    the mechanism question ("is the unlearning bill the prior's scale?")
    is answered by scalars, and scalars stay interpretable in the report.
    """

    def __init__(self, adjacency: Tensor, sigma: float, n_plus_target: int):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.sigma = sigma
        self.n_plus_target = n_plus_target
        self.gate_budget = nn.Parameter(torch.tensor(1.0))
        self.gate_field = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_masked: Tensor) -> Tensor:
        budget_logit, energy_term, interior = budget_tilt_components(
            x_masked, self.adjacency, self.sigma, self.n_plus_target
        )
        gated = (self.gate_budget * budget_logit[:, None]
                 + self.gate_field * energy_term)
        boundary_delta = budget_logit[:, None].expand(-1, x_masked.shape[1])
        return torch.where(interior[:, None], gated, boundary_delta)


def feasibility_clamped_p_plus(
    p_plus: Tensor, budget: Tensor, masked_count: Tensor
) -> Tensor:
    """Generation-time feasibility guard: b = 0 forbids +1, b = m forces it.

    This is the reference process's own mechanism (the budget lives in the
    species draw, never the site clock), applied to the learned conditional
    — it is what makes the gate's G0 structural. It is NOT the
    mask-and-renormalise pathology: the rollout log-probability records the
    CLAMPED probabilities actually sampled from, so the importance weights
    stay consistent with the rollout law (the pathology is computing
    weights with one law while sampling from another, pinned by
    tests/test_budget_masked_reference.py::
    test_mask_and_renormalise_correction_is_trajectory_dependent).
    Asymptotically a no-op: the WDCE minimiser is itself a delta at
    boundary contexts.
    """
    p_plus = torch.where(budget <= 0, torch.zeros_like(p_plus), p_plus)
    return torch.where(budget >= masked_count, torch.ones_like(p_plus),
                       p_plus)


def rollout_budget_masked(
    logit_diff_fn,
    n_rollouts: int,
    n_sites: int,
    n_plus_target: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Generate terminals by sequential revelation; return
    (terminals, rollout_log_prob).

    The embedded jump chain of the controlled CTMC: the clock gamma(t)
    cancels between reference and control (the verified tilts-sum-to-one
    property — control re-routes WHICH species is revealed, never how fast
    sites reveal), so the terminal law depends only on the jump chain and
    the discrete loop below IS the sampler; no Euler error term exists to
    tune, unlike the flip/swap CTMC path.

    rollout_log_prob accumulates log q(species | state) of the SPECIES
    draws only. The uniform site-choice factors (1/m per step, 1/d! per
    trajectory) are identical for every trajectory, and cancel against the
    same factors in the reference measure inside the importance weight —
    dropping them here and in the weight is the same batch-softmax
    invariance that drops the reference constants (verified in
    tests/test_budget_wdce.py::
    test_reference_log_weight_terms_shift_all_trajectories_equally).
    The trajectory importance weight is then
        log w = log p_tilde(X_1) - rollout_log_prob   (+ constants),
    with log p_tilde the unnormalised target log-density (sigma * x^T A x).

    Gradient flow is governed by the CALLER's grad mode: the body takes
    no stance, so a caller under torch.no_grad() gets the cheap detached
    rollout (WDCE's sampling step), while a caller in default grad mode
    gets rollout_log_prob differentiable through the model (what F_LV
    needs — it differentiates the trajectory RN derivative). The DRAWS
    themselves are constants either way: this is the paper's v = u-bar
    convention (gradient-free sampling measure), not a REINFORCE term.
    """
    device = generator.device        # CPU and CUDA generators both carry it
    x_masked = torch.zeros(n_rollouts, n_sites, device=device)
    rollout_log_prob = torch.zeros(n_rollouts, device=device)
    rows = torch.arange(n_rollouts, device=device)
    for step in range(n_sites):
        masked_count, budget = masked_count_and_budget(
            x_masked, n_plus_target
        )
        # uniform masked site per row: Gumbel-argmax over masked positions
        noise = torch.rand(
            n_rollouts, n_sites, generator=generator, device=device
        ).masked_fill(x_masked != 0.0, -1.0)
        site = noise.argmax(dim=1)

        logit = logit_diff_fn(x_masked).gather(1, site[:, None])[:, 0]
        p_plus = feasibility_clamped_p_plus(
            torch.sigmoid(logit), budget, masked_count
        )
        draw_plus = (
            torch.rand(n_rollouts, generator=generator, device=device)
            < p_plus
        )
        # log q via logsigmoid for saturation safety; clamped rows are
        # forced draws with q = 1, i.e. log q = 0
        interior = (budget > 0) & (budget < masked_count)
        log_q = torch.where(draw_plus, logsigmoid(logit),
                            logsigmoid(-logit))
        rollout_log_prob += torch.where(
            interior, log_q, torch.zeros_like(log_q)
        )
        x_masked[rows, site] = torch.where(draw_plus, 1.0, -1.0)
    return x_masked, rollout_log_prob


def wdce_cross_entropy(
    logit_diff_fn,
    terminals: Tensor,
    normalised_weights: Tensor,
    n_replicates: int,
    generator: torch.Generator,
    context_loss_weight=None,
) -> Tensor:
    """The constrained WDCE loss (the paper's Eq. (16) on the fibre).

        F_WDCE^c = sum_k w_k * (1/R) sum_r sum_{d masked in x~_kr}
                       -log s_theta(x~_kr)_{d, X_k^d},

    with w_k the batch-softmax importance weights (detached: they estimate
    the target measure, they are not a differentiation path — self-
    normalised IS, their Eq. (16)), and x~_kr the r-th corruption of
    terminal X_k: lambda ~ U(0,1) per replicate, each site masked
    independently with probability lambda. That corruption is the
    UNCONSTRAINED kernel deliberately — the bridge conditional of the
    budget-masked reference given the terminal is exactly mu_lambda
    (order-assignment independence via the trajectory constant, verified in
    tests/test_budget_wdce.py); a budget-aware corruption would be wrong,
    not conservative. w(lambda) = 1: the minimiser is invariant to the
    corruption-size weight (also verified), so the simplest choice is the
    defensible one. Empty masks contribute an empty sum, as in Eq. (4).

    The per-site term is a numerically stable binary cross-entropy in the
    logit difference: -log s(+1) = softplus(-z), -log s(-1) = softplus(z).

    context_loss_weight (optional): eta(x~) -> (B*R,) positive weights, a
    function of the CORRUPTED CONTEXT alone (e.g. the near-boundary boost
    1 + kappa*1[b in {1, m-1}], with b and m read off the context). Because
    every completion coefficient at a fixed context scales equally, the
    population minimiser is unchanged (the same argument as the w(lambda)
    freedom; pinned by tests/test_budget_wdce.py::
    test_minimiser_is_invariant_to_context_dependent_loss_weights) — the
    weight reallocates GRADIENT between contexts, nothing else. It is
    normalised to batch-mean 1 so the loss scale (and the effective
    learning rate) is comparable across boost settings. A weight that read
    the TERMINAL would not be minimiser-safe; the signature only exposes
    the corrupted context to make that mistake impossible.
    """
    n_terminals, n_sites = terminals.shape
    device = terminals.device        # generator must live on the same device
    replicated = terminals.repeat_interleave(n_replicates, dim=0)
    corruption_level = torch.rand(
        n_terminals * n_replicates, 1, generator=generator, device=device
    )
    corruption_mask = (
        torch.rand(
            n_terminals * n_replicates, n_sites, generator=generator,
            device=device,
        ) < corruption_level
    )
    corrupted = replicated.masked_fill(corruption_mask, 0.0)

    logit = logit_diff_fn(corrupted)
    site_nll = torch.where(
        replicated == 1.0, softplus(-logit), softplus(logit)
    )
    per_replicate = (site_nll * corruption_mask.float()).sum(dim=1)
    if context_loss_weight is not None:
        with torch.no_grad():
            eta = context_loss_weight(corrupted)
            eta = eta / eta.mean()
        per_replicate = per_replicate * eta
    per_terminal = per_replicate.view(n_terminals, n_replicates).mean(dim=1)
    return (normalised_weights.detach() * per_terminal).sum()


def log_variance_loss(
    logit_diff_fn,
    target_log_prob_fn,
    n_rollouts: int,
    n_sites: int,
    n_plus_target: int,
    generator: torch.Generator,
) -> Tensor:
    """The constrained F_LV (the paper's Eq. (10) with v = u-bar, via the
    Eq. (15) simplification): the batch variance of the trajectory
    log-RN-derivative,

        F_LV^c = Var_batch( log p_tilde(X_1) - rollout_log_prob ).

    Why this transfers to the fibre with no extra terms: the exact
    log dP*/dP^u adds the reference's trajectory constant and the
    uniform-on-fibre base constant to the expression above, and a
    variance is invariant to additive constants — the same cancellation
    the batch softmax buys WDCE, bought here by Var instead. The paper's
    4x4 case studies rank F_LV their strongest objective at this size
    (their Tabs. 2 and 4: at beta_critical ESS 0.9809 / path-KL 0.0083
    vs WDCE's 0.9644 / 0.0177), which is why the gate plan pre-scoped it
    as the robustness arm. Cost note: unlike WDCE this differentiates
    through every species draw of the rollout (the paper's stated reason
    to prefer WDCE at 16x16 scale); at d = 16 the graph is 16 tiny-MLP
    calls deep and fits trivially.

    Trained UNCLAMPED: like the WDCE loss, the objective sees the raw
    parameterised conditional; the feasibility clamp remains a
    generation-time guard (boundary steps contribute log q = 0 with zero
    gradient either way, since the clamp forces q = 1 there).

    Returns (loss, terminals, detached log-RN) so a training loop can
    log ESS/feasibility off the SAME rollout that carried the gradient —
    F_LV consumes one rollout per step where WDCE consumes rollout + a
    separate corruption pass.
    """
    terminals, rollout_log_prob = rollout_budget_masked(
        logit_diff_fn, n_rollouts, n_sites, n_plus_target, generator
    )
    log_rn = target_log_prob_fn(terminals) - rollout_log_prob
    return log_rn.var(), terminals.detach(), log_rn.detach()
