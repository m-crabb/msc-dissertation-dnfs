"""Budget-masked masked-diffusion sampler on the fixed-composition fibre.

The masked-diffusion analogue of the swap CTMC: rather than restricting the
move set, the reference reveals a uniformly chosen masked site and draws
species +1 with probability b/m (b = remaining +1 budget, m = masked
sites). Its terminal law is uniform on the fibre and every trajectory
carries the same reference constant N_+!(d-N_+)!/d! (trajectory-constant
lemma, tests/test_budget_masked_reference.py), so all reference terms
cancel under self-normalisation.

Verified facts (the tests are the durable record of the derivations):

1. Constrained Lemma 3: the optimal control unmasks site i to species s at
   rate gamma(t) * Pr_pi(X^i = s | X^UM), the constrained target's masked
   conditional; the urn factor b/m cancels the value-function ratio
   C(m,b)/C(m-1,b-1) = m/b (tests/test_budget_preconditioner.py).
2. The budget-tilted preconditioner V0 approximates it in closed form:
   logit(+1) - logit(-1) = log(b/(m-b)) + 4*sigma*f_i, masked neighbours
   imputed at the urn mean (2b-m)/m. Exact at sigma = 0, at m = 1 and at
   exhausted budgets.
3. WDCE transfers unchanged: corruption is the unconstrained kernel
   (order-assignment independence) and the population minimiser at every
   context is the Lemma-3 conditional (tests/test_budget_wdce.py).

Conventions: masked states are float tensors with +1.0/-1.0 spins and 0.0
at masked sites (0 doubles as the zero-imputation value, so the field is a
single matmul); adjacency is IsingTarget's symmetric 0/1 matrix, giving the
single-site tilt E(+1) - E(-1) = 4*(A x)_i on the x^T A x energy.
"""

import torch
from torch import Tensor, nn
from torch.nn.functional import logsigmoid, softplus

# log(b/(m-b)) diverges at b = 0 or b = m; the closed form and the exact
# delta agree in the limit, so this branch is float safety. 30 saturates a
# float32 sigmoid (~1e-13) and leaves room for a trunk logit without overflow.
BOUNDARY_LOGIT = 30.0


def masked_count_and_budget(
    x_masked: Tensor, n_plus_target: int
) -> tuple[Tensor, Tensor]:
    """Per-row (m, b): masked-site count and remaining +1 budget.

    Pure functions of the state, so nothing falls out of sync with the
    reveals; WDCE corruption masks arbitrary subsets, where incremental
    bookkeeping has no meaning.
    """
    masked_count = (x_masked == 0.0).sum(dim=1)
    budget = n_plus_target - (x_masked == 1.0).sum(dim=1)
    return masked_count, budget


def masked_state_features(x_masked: Tensor) -> Tensor:
    """Three-channel one-hot encoding (+1 / -1 / masked), shape (B, 3d).

    b and m are computable from it, so the trunk can learn the budget
    structure; the no-preconditioner ablation measures how much it must.
    """
    return torch.cat(
        [x_masked == 1.0, x_masked == -1.0, x_masked == 0.0], dim=1
    ).float()


class MaskedConditionalNet(nn.Module):
    """Minimal trunk Phi_theta for the masked conditional, MDNS-style.

    Outputs a per-site logit difference logit(+1) - logit(-1); the sampler's
    conditional is sigmoid(Phi_theta(x) + P(x)) with P the preconditioner,
    the paper's softmax(network + preconditioner) reduced to two species.
    The final layer is zero-initialised so step 0 is the preconditioner's
    law, which the Fig.-10 ablation compares across modes.
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
    """The mode-defining logit offset P(x), shape (B, d); rows are valid at
    masked sites only.

    "budget_tilted": V0,
        log(b/(m-b)) + 4*sigma*f_i,  f_i = (A x_imputed)_i,
    masked neighbours imputed at the urn mean (2b-m)/m, the imputation that
    is unbiased under the urn's exchangeable marginal; boundary budgets take
    the pseudo-infinite branch (exact deltas).
    "unconstrained": the paper's App. D.4 form, 4*sigma*f_i with masked
    neighbours at zero and no budget term.
    "none": zero (the Fig.-10 no-preconditioning mirror).
    Not the hypothesis-conditioned imputation V1: ~10% lower mean error but
    a hypothesis-dependent field pass, and the same exactness limits as V0.
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
    logit[boundary] = budget_logit[boundary][:, None].expand(-1, x_masked.shape[1])
    return logit


def budget_tilt_components(
    x_masked: Tensor, adjacency: Tensor, sigma: float, n_plus_target: int
) -> tuple[Tensor, Tensor, Tensor]:
    """V0's two logit terms separately: (budget_logit (B,), energy_term
    (B, d), interior (B,) bool).

    budget_logit is log(b/(m-b)) on interior rows and the pseudo-infinite
    delta on boundary rows; energy_term is 4*sigma*f with urn-mean
    imputation. Split out so `GatedBudgetTiltOffset` can scale the two
    mechanisms independently.
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
    field = x_masked @ adjacency + urn_mean[:, None] * (masked_indicator @ adjacency)
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
    """V0 with learnable scales on its two mechanisms.

        P(x) = gate_budget * log(b/(m-b)) + gate_field * energy_term
        (interior rows; boundary rows keep the ungated exact deltas)

    Both gates initialise at 1.0, so step 0 is the V0 law. The first 4x4
    runs showed the trunk partially cancelling V0's near-boundary opinions
    (the unconstrained preconditioner ends better there); two scalars make
    that cancellation a 2-parameter descent. The boundary branch stays
    ungated because it is exact at any coupling.
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
        gated = self.gate_budget * budget_logit[:, None] + self.gate_field * energy_term
        boundary_delta = budget_logit[:, None].expand(-1, x_masked.shape[1])
        return torch.where(interior[:, None], gated, boundary_delta)


def feasibility_clamped_p_plus(
    p_plus: Tensor, budget: Tensor, masked_count: Tensor
) -> Tensor:
    """Generation-time feasibility guard: b = 0 forbids +1, b = m forces it.

    The reference process's own mechanism (the budget lives in the species
    draw, never the site clock) applied to the learned conditional, which
    makes on-fibre generation structural. Not the mask-and-renormalise
    pathology: the rollout log-probability records the clamped probabilities
    actually sampled from, so the weights stay consistent with the rollout
    law (tests/test_budget_masked_reference.py). Asymptotically a no-op: the
    WDCE minimiser is itself a delta at boundary contexts.
    """
    p_plus = torch.where(budget <= 0, torch.zeros_like(p_plus), p_plus)
    return torch.where(budget >= masked_count, torch.ones_like(p_plus), p_plus)


def rollout_budget_masked(
    logit_diff_fn,
    n_rollouts: int,
    n_sites: int,
    n_plus_target: int | None,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Generate terminals by sequential revelation; return
    (terminals, rollout_log_prob).

    n_plus_target = None switches the budget machinery off: the reference
    is the paper's unconstrained masked diffusion (uniform site, species +1
    with probability 1/2), no feasibility clamp, every draw counted. The
    trajectory constant 1/C(d, N_+) becomes (1/2)^d, the uniform base
    measure on {-1,+1}^d, so it cancels the same way the fibre constant
    cancels against uniform-on-the-fibre; logmeanexp(log w) then estimates
    the corresponding log Z (full-space or slice).
    Pinned by tests/test_budget_masked_sampler.py.

    This is the embedded jump chain of the controlled CTMC: gamma(t) cancels
    between reference and control (tilts sum to one; control re-routes
    which species is revealed, never how fast), so the terminal law depends
    only on the jump chain and there is no Euler error to tune.

    rollout_log_prob accumulates log q(species | state) of the species draws
    only; the uniform site-choice factors (1/m per step, 1/d! per
    trajectory) are identical across trajectories and cancel in the weight
    (tests/test_budget_wdce.py). The trajectory importance weight is
        log w = log p_tilde(X_1) - rollout_log_prob   (+ constants),
    with log p_tilde the unnormalised target log-density (sigma * x^T A x).

    Grad mode is the caller's: under torch.no_grad() the rollout is detached
    (WDCE's sampling step); in default mode rollout_log_prob is
    differentiable through the model (F_LV). The draws themselves are
    constants either way: the paper's v = u-bar convention, not REINFORCE.
    """
    device = generator.device  # CPU and CUDA generators both carry it
    x_masked = torch.zeros(n_rollouts, n_sites, device=device)
    rollout_log_prob = torch.zeros(n_rollouts, device=device)
    rows = torch.arange(n_rollouts, device=device)
    for step in range(n_sites):
        # uniform masked site per row: Gumbel-argmax over masked positions
        noise = torch.rand(
            n_rollouts, n_sites, generator=generator, device=device
        ).masked_fill(x_masked != 0.0, -1.0)
        site = noise.argmax(dim=1)

        logit = logit_diff_fn(x_masked).gather(1, site[:, None])[:, 0]
        if n_plus_target is None:
            p_plus = torch.sigmoid(logit)
        else:
            masked_count, budget = masked_count_and_budget(x_masked, n_plus_target)
            p_plus = feasibility_clamped_p_plus(
                torch.sigmoid(logit), budget, masked_count
            )
        draw_plus = torch.rand(n_rollouts, generator=generator, device=device) < p_plus
        # log q via logsigmoid for saturation safety; clamped rows are forced
        # draws with q = 1, so log q = 0 (the unconstrained reference never clamps)
        log_q = torch.where(draw_plus, logsigmoid(logit), logsigmoid(-logit))
        if n_plus_target is None:
            rollout_log_prob += log_q
        else:
            interior = (budget > 0) & (budget < masked_count)
            rollout_log_prob += torch.where(interior, log_q, torch.zeros_like(log_q))
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

    with w_k the detached batch-softmax importance weights (self-normalised
    IS, Eq. (16)) and x~_kr the r-th corruption of terminal X_k: lambda ~
    U(0,1) per replicate, each site masked independently with probability
    lambda. The corruption is the unconstrained kernel deliberately: the
    bridge conditional of the budget-masked reference given the terminal is
    exactly mu_lambda (tests/test_budget_wdce.py), so a budget-aware
    corruption would be wrong, not conservative. w(lambda) = 1, since the
    minimiser is invariant to it. Empty masks contribute an empty sum, as
    in Eq. (4).

    Per-site term: -log s(+1) = softplus(-z), -log s(-1) = softplus(z).

    context_loss_weight (optional): eta(x~) -> (B*R,) positive weights, a
    function of the corrupted context alone (e.g. the near-boundary boost
    1 + kappa*1[b in {1, m-1}]). Every completion coefficient at a fixed
    context scales equally, so the population minimiser is unchanged and
    the weight only reallocates gradient between contexts
    (tests/test_budget_wdce.py). Normalised to batch-mean 1 so the loss
    scale is comparable across boost settings. A weight that read the
    terminal would not be minimiser-safe, so the signature exposes only
    the corrupted context.
    """
    n_terminals, n_sites = terminals.shape
    device = terminals.device  # generator must live on the same device
    replicated = terminals.repeat_interleave(n_replicates, dim=0)
    corruption_level = torch.rand(
        n_terminals * n_replicates, 1, generator=generator, device=device
    )
    corruption_mask = (
        torch.rand(
            n_terminals * n_replicates,
            n_sites,
            generator=generator,
            device=device,
        )
        < corruption_level
    )
    corrupted = replicated.masked_fill(corruption_mask, 0.0)

    logit = logit_diff_fn(corrupted)
    site_nll = torch.where(replicated == 1.0, softplus(-logit), softplus(logit))
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
    n_plus_target: int | None,
    generator: torch.Generator,
) -> Tensor:
    """The constrained F_LV (the paper's Eq. (10) with v = u-bar, via the
    Eq. (15) simplification): the batch variance of the trajectory
    log-RN-derivative,

        F_LV^c = Var_batch( log p_tilde(X_1) - rollout_log_prob ).

    Transfers to the fibre with no extra terms: the exact log dP*/dP^u adds
    the reference trajectory constant and the uniform-on-fibre constant, and
    a variance is invariant to additive constants. The paper's 4x4 case
    studies rank F_LV strongest at this size (Tabs. 2 and 4: at
    beta_critical ESS 0.9809 / path-KL 0.0083 vs WDCE's 0.9644 / 0.0177),
    so it is carried as the robustness objective. Unlike WDCE it
    differentiates through every species draw of the rollout (the paper's
    reason to prefer WDCE at 16x16).

    Trained unclamped, like WDCE; the feasibility clamp is a generation-time
    guard (boundary steps contribute log q = 0 with zero gradient either way).

    Returns (loss, terminals, detached log-RN) so a training loop can log
    ESS/feasibility off the same rollout that carried the gradient.
    """
    terminals, rollout_log_prob = rollout_budget_masked(
        logit_diff_fn, n_rollouts, n_sites, n_plus_target, generator
    )
    log_rn = target_log_prob_fn(terminals) - rollout_log_prob
    return log_rn.var(), terminals.detach(), log_rn.detach()
