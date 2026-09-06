"""Optimisers shared by the single-site and swap training loops."""

import torch


class StableAdamW(torch.optim.Optimizer):
    """AdamW with Adafactor-style per-tensor update clipping (StableAdamW).

    Wortsman et al. 2023 ("Stable and low-precision training..."), Algorithm
    1: standard AdamW moments, but the step for each parameter TENSOR is
    scaled by 1/max(1, RMS_t/clip) with RMS_t = sqrt(mean(g_t^2 / v_hat_t)),
    i.e. the root-mean-square of the would-be Adam update ratio. Why this and
    not a raw gradient-norm clip: the swap loss is an unnormalised sum over
    d(d-1)/2 pairs, so a fixed grad_clip_max_norm carries EXTENSIVE units and
    silently changes meaning with lattice size (spike guard at d=64,
    permanent normalisation at d=256 — the 16x16 divergence). RMS_t is
    unit-free: it compares the gradient to the optimiser's own noise scale
    v_hat, so clip=1.0 means the same thing at every d. The decoupled weight
    decay uses the same scaled lr, per the paper. Guarded against: the
    documented rescale-not-skip runaway, where a degenerate batch buys the
    largest permitted step in its own noise direction — here a spike inflates
    its own RMS_t and is damped tensor-wise instead of renormalised globally.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        clip_threshold=1.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                v_hat = exp_avg_sq / bias2
                # Per-tensor update-ratio RMS; eps^2 floor keeps fresh moments
                # from dividing by ~0 on the very first steps.
                rms = (grad.pow(2) / v_hat.clamp_min(group["eps"] ** 2)).mean().sqrt()
                lr_eff = group["lr"] / max(1.0, (rms / group["clip_threshold"]).item())
                if group["weight_decay"] != 0:
                    p.mul_(1 - lr_eff * group["weight_decay"])
                denom = v_hat.sqrt().add_(group["eps"])
                p.addcdiv_(exp_avg / bias1, denom, value=-lr_eff)
        return loss
