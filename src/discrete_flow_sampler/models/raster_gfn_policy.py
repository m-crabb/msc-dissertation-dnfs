"""Autoregressive raster-order policy for the GFlowNet comparator.

The comparator's construction chain assigns lattice sites in raster order,
so a state is a prefix x_{<i} and the forward policy is a product of per-site
Bernoulli conditionals

    q_theta(x) = prod_i P_F(x_i | x_{<i}),

with the exactly-N_A composition constraint enforced by count masking: after
placing n_up up-spins with r sites remaining, the conditional is forced to -1
once n_up == N_A and forced to +1 once N_A - n_up == r. Every trajectory is
feasible by construction ("the action space is set up so that infeasible
states are never reachable"), which is the route this comparator exists to
test against the CTMC-level projection.

Why a causal transformer and a FIXED order: scoring a full configuration
computes all d conditionals in ONE causal forward pass (O(d^2) attention),
where a general GFlowNet framework's state->logits map re-encodes each of the
d prefixes separately (O(d^3)) — the difference between a runnable and an
unrunnable 16x16 cell. Fixed raster order also gives each state exactly one
parent, so P_B == 1 and both the flow-underdetermination problem and the
learned-backward-policy design question disappear.

Failure mode guarded here: the classic AR off-by-one, where the feature that
predicts x_i has already seen x_i. Token i is the embedding of x_{i-1} (BOS
at i=0), so position i attends only to sites < i; the sequential sampler and
the parallel scorer share `_encode`, and a test pins their agreement.
"""

import torch
from torch import nn
from torch.nn import functional as F

_BOS_TOKEN_ID = 2  # spins map to token ids {0, 1}; 2 marks sequence start
_NEG_INF = float("-inf")


class _CausalBlock(nn.Module):
    """Pre-norm transformer block with causally masked self-attention."""

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normed = self.attention_norm(h)
        attended, _ = self.attention(
            normed, normed, normed, attn_mask=causal_mask, need_weights=False
        )
        h = h + attended
        return h + self.mlp(self.mlp_norm(h))

    def forward_step(self, h_step: torch.Tensor, cache: dict) -> torch.Tensor:
        """One-token forward against cached keys/values (KV-cache path).

        Reuses `forward`'s in_proj/out_proj weights at the newest position.
        The cache holds only positions <= the current one, so no mask is
        needed. Attention costs O(L) per step instead of O(L^2) for a full
        re-encode: O(d^2) over a d-step rollout instead of O(d^3).
        """
        attention = self.attention
        batch, _, hidden_dim = h_step.shape
        n_heads = attention.num_heads
        head_dim = hidden_dim // n_heads

        normed = self.attention_norm(h_step)
        query, key, value = F.linear(
            normed, attention.in_proj_weight, attention.in_proj_bias
        ).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, 1, n_heads, head_dim).transpose(1, 2)

        length = cache["length"]
        cache["k"][:, :, length] = split_heads(key).squeeze(2)
        cache["v"][:, :, length] = split_heads(value).squeeze(2)
        cache["length"] = length + 1

        attended = F.scaled_dot_product_attention(
            split_heads(query),
            cache["k"][:, :, : length + 1],
            cache["v"][:, :, : length + 1],
        )
        attended = attended.transpose(1, 2).reshape(batch, 1, hidden_dim)
        h_step = h_step + attention.out_proj(attended)
        return h_step + self.mlp(self.mlp_norm(h_step))


class RasterGFNPolicy(nn.Module):
    """Count-masked AR policy over the fixed-composition slice.

    Heads on the shared causal features:
      * policy head    -> logit for P(x_i = +1 | x_{<i})  (before masking);
      * `log_z`        -> scalar log-partition estimate for the TB objective
                          (Malkin et al. 2022: at the TB optimum it equals
                          the slice log Z, so it is also a free estimator);
      * flow head      -> log F_res(s_i), the forward-looking flow RESIDUAL
                          of prefix s_i (Pan et al. 2023). The full log-flow
                          is log F(s_i) = log F_res(s_i) + partial_energy(s_i)
                          and the terminal residual is 0 by convention since
                          a complete prefix's partial energy IS log p_tilde.
    """

    def __init__(
        self,
        D: int,
        n_plus_target: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        with_flow_head: bool = False,
        standalone_flow_head: bool = False,
    ):
        super().__init__()
        self.D = D
        self.d = D * D
        self.n_plus_target = n_plus_target
        self.standalone_flow_head = standalone_flow_head

        self.token_embedding = nn.Embedding(3, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(self.d, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            _CausalBlock(hidden_dim, n_heads) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, 1)
        self.log_z = nn.Parameter(torch.zeros(()))
        # Two flow parameterisations. The shared-trunk readout (the
        # original) reads log F_res off the causal features — cheapest, but
        # its gradients flow INTO the policy trunk, and the flow-lr
        # sweeps surfaced the failure mode: a fast flow readout absorbs DB
        # residuals and SHIELDS the policy from its own gradient signal.
        # The standalone module is the torchgfn convention (a separate
        # ScalarEstimator over the state, verified in reference source):
        # it decouples flow gradients from the trunk entirely, at a
        # declared parameter cost (a small MLP over the 3-way one-hot
        # prefix encoding; hidden = trunk hidden_dim, 2 hidden layers —
        # a sizing choice kept small to bound the delta).
        if not with_flow_head:
            self.flow_head = None
        elif standalone_flow_head:
            self.flow_head = nn.Sequential(
                nn.Linear(3 * self.d, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.flow_head = nn.Linear(hidden_dim, 1)

        # True above the diagonal = position i may not attend to sites >= i.
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(self.d, self.d, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    # -- shared encoder ----------------------------------------------------

    def _encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """(B, L) token ids -> (B, L, hidden); feature i has seen sites < i."""
        length = token_ids.shape[1]
        h = self.token_embedding(token_ids) + self.position_embedding[:length]
        mask = self._causal_mask[:length, :length]
        for block in self.blocks:
            h = block(h, mask)
        return self.final_norm(h)

    def _new_kv_caches(self, batch: int) -> list[dict]:
        """Preallocated per-block KV caches for a d-step cached rollout."""
        hidden_dim = self.position_embedding.shape[1]
        n_heads = self.blocks[0].attention.num_heads
        head_dim = hidden_dim // n_heads
        device = self.position_embedding.device
        return [
            {
                "k": torch.zeros(batch, n_heads, self.d, head_dim, device=device),
                "v": torch.zeros(batch, n_heads, self.d, head_dim, device=device),
                "length": 0,
            }
            for _ in self.blocks
        ]

    def _encode_step(
        self, token_ids_step: torch.Tensor, site: int, caches: list[dict]
    ) -> torch.Tensor:
        """(B,) token ids at position `site` -> (B, hidden) causal feature,
        advancing the per-block caches. Must equal `_encode(...)[:, site]`
        (pinned by test_kv_cache_step_features_match_full_encode)."""
        h = (
            self.token_embedding(token_ids_step) + self.position_embedding[site]
        ).unsqueeze(1)
        for block, cache in zip(self.blocks, caches):
            h = block.forward_step(h, cache)
        return self.final_norm(h).squeeze(1)

    def _shifted_token_ids(self, spins: torch.Tensor) -> torch.Tensor:
        """[BOS, x_0, ..., x_{d-2}]: the AR shift, so feature i predicts x_i."""
        spin_ids = (spins > 0).long()
        bos = torch.full(
            (spins.shape[0], 1), _BOS_TOKEN_ID, dtype=torch.long, device=spins.device
        )
        return torch.cat([bos, spin_ids[:, :-1]], dim=1)

    # -- count masking -----------------------------------------------------

    def _forced_moves(self, n_up_before: torch.Tensor, site_index) -> tuple:
        """Boolean (force_up, force_down) given up-count before each site.

        force_down uses >= rather than == so that scoring an OFF-slice state
        (too many up-spins) yields -inf instead of silently renormalising.
        The two can never both hold on a reachable prefix: they would need
        N_A - n_up == remaining and n_up == N_A with remaining >= 1.
        """
        sites_remaining = self.d - site_index
        force_up = (self.n_plus_target - n_up_before) == sites_remaining
        force_down = n_up_before >= self.n_plus_target
        return force_up, force_down

    def _masked_chosen_log_probs(
        self, spins: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """(B, d) log P_F of the spin actually present, under the count mask."""
        up = (spins > 0).float()
        n_up_before = up.cumsum(dim=-1) - up
        site_index = torch.arange(self.d, device=spins.device)
        force_up, force_down = self._forced_moves(n_up_before, site_index)

        log_p_up = F.logsigmoid(logits)
        log_p_down = F.logsigmoid(-logits)
        log_p_up = torch.where(force_up, 0.0, log_p_up)
        log_p_up = torch.where(force_down, _NEG_INF, log_p_up)
        log_p_down = torch.where(force_down, 0.0, log_p_down)
        log_p_down = torch.where(force_up, _NEG_INF, log_p_down)
        return torch.where(spins > 0, log_p_up, log_p_down)

    # -- scoring (one causal pass) ----------------------------------------

    def site_log_probs(self, spins: torch.Tensor) -> torch.Tensor:
        """(B, d) masked per-site conditionals, one forward pass."""
        features = self._encode(self._shifted_token_ids(spins))
        logits = self.policy_head(features).squeeze(-1)
        return self._masked_chosen_log_probs(spins, logits)

    def log_prob(self, spins: torch.Tensor) -> torch.Tensor:
        """log q_theta(x), (B,). Off-slice states score -inf (mask violated)."""
        return self.site_log_probs(spins).sum(dim=-1)

    def _standalone_flow_residuals(self, spins: torch.Tensor) -> torch.Tensor:
        """(B, d) residuals from the standalone module: entry i is
        log F_res of the prefix BEFORE site i (sites < i assigned, the rest
        an explicit 'unassigned' class), matching the loss's alignment
        exactly. All d prefixes of each sample are encoded as a 3-way
        one-hot over sites — (B, d, 3d) — and scored in one MLP batch."""
        batch, d = spins.shape
        token_ids = (spins > 0).long()  # (B, d) in {0, 1}
        site = torch.arange(d, device=spins.device)
        assigned = site[None, :] < site[:, None]  # (d_prefix, d_site)
        prefix_ids = torch.where(
            assigned[None, :, :], token_ids[:, None, :], 2
        )  # (B, d, d); 2 = unassigned
        one_hot = torch.nn.functional.one_hot(prefix_ids, 3).float()
        return self.flow_head(one_hot.reshape(batch, d, 3 * d)).squeeze(-1)

    def site_log_probs_and_flow_residuals(self, spins: torch.Tensor) -> tuple:
        """Both FL-DB ingredients; one shared encoder pass for the policy,
        and the flow from whichever parameterisation the cell declares."""
        if self.flow_head is None:
            raise ValueError(
                "policy was built with with_flow_head=False; the FL-DB arm "
                "needs RasterGFNPolicy(..., with_flow_head=True)"
            )
        features = self._encode(self._shifted_token_ids(spins))
        logits = self.policy_head(features).squeeze(-1)
        if self.standalone_flow_head:
            flow_residuals = self._standalone_flow_residuals(spins)
        else:
            flow_residuals = self.flow_head(features).squeeze(-1)
        return self._masked_chosen_log_probs(spins, logits), flow_residuals

    # -- sequential sampling ----------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n: int,
        epsilon: float = 0.0,
        generator: torch.Generator | None = None,
        kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw n slice configurations; returns (spins, log q_theta(spins)).

        epsilon mixes the POLICY with a uniform behaviour distribution
        (standard GFN exploration); the returned log-probability is always
        the policy's own, which is what the TB loss needs — with epsilon > 0
        training is off-policy, which TB tolerates without importance weights
        (the one distinctly-GFN property this comparator keeps).

        The mask binds the behaviour policy too, so exploration cannot leave
        the slice. The default path caches per-block keys/values, so a
        rollout costs O(d^2) attention; `kv_cache=False` keeps the naive
        one-prefix-re-encode-per-site path (O(d^3)) whose only remaining
        job is pinning the cache as an exact rewrite
        (test_sample_with_and_without_kv_cache_agree).
        """
        device = self.position_embedding.device
        spins = torch.zeros(n, self.d, device=device)
        log_q = torch.zeros(n, device=device)
        n_up = torch.zeros(n, device=device)
        caches = self._new_kv_caches(n) if kv_cache else None

        for site in range(self.d):
            if kv_cache:
                if site == 0:
                    token_ids_step = torch.full(
                        (n,), _BOS_TOKEN_ID, dtype=torch.long, device=device
                    )
                else:
                    token_ids_step = (spins[:, site - 1] > 0).long()
                feature = self._encode_step(token_ids_step, site, caches)
            else:
                token_ids = self._shifted_token_ids(spins[:, : site + 1])
                feature = self._encode(token_ids)[:, -1]
            logit = self.policy_head(feature).squeeze(-1)
            force_up, force_down = self._forced_moves(n_up, site)

            probability_up = torch.sigmoid(logit)
            behaviour_up = (1.0 - epsilon) * probability_up + epsilon * 0.5
            behaviour_up = torch.where(force_up, 1.0, behaviour_up)
            behaviour_up = torch.where(force_down, 0.0, behaviour_up)

            uniform = torch.rand(n, device=device, generator=generator)
            chose_up = uniform < behaviour_up
            spins[:, site] = torch.where(chose_up, 1.0, -1.0)

            log_p_up = F.logsigmoid(logit)
            log_p_down = F.logsigmoid(-logit)
            log_p_up = torch.where(force_up, 0.0, log_p_up)
            log_p_down = torch.where(force_down, 0.0, log_p_down)
            log_q += torch.where(chose_up, log_p_up, log_p_down)
            n_up += chose_up.float()

        return spins, log_q
