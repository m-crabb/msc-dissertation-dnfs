"""Losses and prefix rewards for the raster-order GFlowNet comparator.

Hand-rolled rather than imported from torchgfn: on the fixed raster chain
P_B == 1 and the losses reduce to a few lines, the enumeration-exact tests in
test_gfn_comparator.py are a stronger correctness authority than a reference
implementation, and torchgfn's state-map abstraction would re-encode every
prefix (O(d^3) attention at d=256, where the causal policy scores all d
conditionals in one O(d^2) pass). torchgfn remains the conventions reference
(its FL-DB lives in DBGFlowNet(forward_looking=True) with intermediate
rewards from env.log_reward, mirrored here as prefix partial energies).
"""

import torch


def raster_prefix_log_reward_increments(target, spins: torch.Tensor) -> torch.Tensor:
    """(B, d) per-site increments of the prefix partial energy.

    The FL-DB intermediate reward is the partial energy of a prefix: the
    target's log p_tilde restricted to bonds with BOTH endpoints assigned.
    Assign each bond to its LARGER raster index; under raster order that
    endpoint is assigned second (periodic wrap bonds included), so the
    increment at flat site m is

        delta_m = 2 * x_m * sum_{j in N(m), j < m} J_mj x_j + bias * x_m,

    the factor 2 matching IsingTarget's convention that x^T J x counts each
    edge twice. The increments telescope exactly:

        sum_m delta_m = x^T J x + bias * sum_i x_i = target.log_prob(x),

    which is the identity the FL-DB terminal convention (zero residual at the
    complete prefix) relies on — a test pins it. Rejected alternative:
    zeroing unassigned sites and recomputing the full quadratic per prefix is
    O(d^3) per batch; this is one (B, d) @ (d, d) matmul.
    """
    lower_bonds = torch.tril(target.J)  # J symmetric, zero diagonal
    assigned_neighbour_field = spins @ lower_bonds.T
    return 2.0 * spins * assigned_neighbour_field + target.bias * spins


def trajectory_balance_loss(
    log_z: torch.Tensor,
    model_log_prob: torch.Tensor,
    target_log_prob: torch.Tensor,
) -> torch.Tensor:
    """TB loss on the raster chain (Malkin et al. 2022, Eq. 14 collapsed).

    With one parent per state P_B == 1, so the trajectory-level balance
    Z * P_F(tau) = R(x) * P_B(tau | x) reduces to

        L_TB = mean_x ( log Z_theta + log q_theta(x) - log p_tilde(x) )^2 .

    At the optimum q_theta = p on the slice and log Z_theta = the slice
    log-partition, so the learned scalar doubles as a free log Z estimator.
    On-policy this has the same expected gradient as reverse-KL variational
    inference (Malkin et al. 2023, Prop. 1) — the mode-seeking failure the
    FL-DB arm and epsilon-exploration exist to check against.
    """
    residual = log_z + model_log_prob - target_log_prob
    return (residual**2).mean()


def forward_looking_db_loss(
    site_log_probs: torch.Tensor,
    reward_increments: torch.Tensor,
    flow_residuals: torch.Tensor,
) -> torch.Tensor:
    """Forward-looking detailed balance on the chain (Pan et al. 2023).

    Writing the state flow as F(s) = exp(partial_energy(s)) * F_res(s), the
    per-step balance F(s_i) P_F(x_i|s_i) = F(s_{i+1}) becomes, in logs,

        log F_res(s_i) + log P_F(x_i|s_i) - delta_i = log F_res(s_{i+1}),

    with delta_i the partial-energy increment of assigning site i and the
    terminal residual log F_res(s_d) == 0 (a complete prefix's partial energy
    IS log p_tilde, so the terminal flow equals the reward with no residual).
    Each of the d residuals carries the LOCAL energy change — the dense
    credit assignment that lets FL-DB survive the hundreds-step trajectories
    where TB's single trajectory-level residual is documented to degrade
    (the reason this arm exists at d=256).

    Shapes: all (B, d); flow_residuals[:, i] is the model's log F_res(s_i)
    for the prefix BEFORE site i is assigned.
    """
    terminal = torch.zeros_like(flow_residuals[:, :1])
    next_flow_residuals = torch.cat([flow_residuals[:, 1:], terminal], dim=1)
    residual = flow_residuals + site_log_probs - reward_increments - next_flow_residuals
    return (residual**2).mean()
