"""bf16 on the TRAINING step: the byte-reduction lever, and its guard rails.

WHY IT EXISTS. The d400 swap step is bandwidth-bound, not FLOP-bound:
profiled at 36% GEMM, with those GEMMs at 8.0 FLOP/byte against the A100's
fp32 ridge point of 10.1. Levers that reduce arithmetic therefore do almost
nothing (TF32 measured -10% time, -0% memory) while levers that reduce
BYTES do a lot. Measured d400 R=3, batch 512, compiled, A100-80GB:

    fp32          0.400 s   63.41 GB
    TF32          0.359 s   63.41 GB
    bf16          0.271 s   41.63 GB      <- this flag
    bf16 + TF32   0.271 s   41.60 GB      <- TF32 adds nothing on top

That last row is the mechanism confirmed rather than the outcome observed:
once the bytes are halved, what remains is not waiting on arithmetic.

THE TWO THINGS THAT MUST NOT MOVE, and why they are tests and not comments:

  * THE IMPORTANCE WEIGHTS. bf16 reaching the target's swap log-ratio would
    make this an ESTIMATOR change, not a speed lever. It does not: the
    closed form is t*sigma*[2(x_j - x_i)(h_i - h_j) - 2(x_j - x_i)^2 A_ij]
    with h = x @ A, and h is a sum over four torus neighbours, so it lives
    in {-4, -2, 0, 2, 4} at EVERY lattice size -- integers bf16 represents
    exactly. Measured drift 0.000e+00 at D=16 and D=20.

  * THE FROZEN EVAL. Every reported number in all three threads is an fp32
    read; the chapter says so. This flag is scoped to the loss update and
    must leave the eval path alone, so a cell that trains in bf16 is still
    EVALUATED in fp32.

Default False, so every archived cell is byte-identical.
"""
import torch

from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _target(D):
    return FixedCompositionIsingTarget(
        D=D, sigma=0.220343, target_composition=0.5
    )


# The fp32 cells the `bf16` twins are read against, one template per rung.
# Both couplings are covered on purpose: the floor pair settled the COST half
# of the precision question (4/4 matched seeds, -21% to -27% end-to-end) and
# the sigma_c pair exists to settle the QUALITY half, which the
# floor could not because every cell sat on the sampling ceiling.
_D400_FP32_TEMPLATES = (
    "H2_d400_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w4",
    "H2_d400_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w4",
)


def _d400_precision_pairs(CONFIGS):
    """Yield (label, fp32_cell, bf16_cell) for every d400 precision twin."""
    for template in _D400_FP32_TEMPLATES:
        for arm in ("thp2", "thp3"):
            name = template.format(arm=arm)
            yield name, CONFIGS[name], CONFIGS[name + "bf16"]


def test_swap_log_ratio_is_bit_identical_under_bf16_autocast():
    """The estimator gate. If this ever fails, the flag stops being a speed
    lever and becomes a change to the importance weights."""
    for D in (8, 16, 20):
        target = _target(D)
        x = target.sample_base(8, device="cpu")
        t = torch.full((8,), 0.5)
        pairs = upper_tri_pairs(D * D, x.device)
        exact = target.swap_log_ratio(x, t, pairs)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            cast = target.swap_log_ratio(x, t, pairs)
        assert cast.dtype is torch.float32, D
        assert (cast - exact).abs().max().item() == 0.0, D


def test_kawasaki_field_is_integer_valued_at_every_lattice_size():
    """The REASON the gate above passes, pinned separately so a change to
    the adjacency convention cannot silently invalidate it: h = x @ A is a
    sum over four torus neighbours, so it is a small integer regardless of
    d, and small integers are exact in bf16."""
    for D in (8, 16, 20):
        target = _target(D)
        field = target.sample_base(16, device="cpu") @ target.A
        assert field.min().item() >= -4.0 and field.max().item() <= 4.0, D
        assert torch.equal(field, field.round()), D


def test_train_autocast_bf16_defaults_off_everywhere():
    """Archived cells must be byte-identical, so the field defaults False
    and no existing cell turns it on. Only the `_w4bf16` d400 probe cells
    and the `_w5bf16` d576 cells (bf16-only at d400+) may carry it."""
    from experiments.constrained_hard_03.configs import CONFIGS

    on = [n for n, c in CONFIGS.items()
          if getattr(c.train, "train_autocast_bf16", False)]
    assert all(n.endswith(("_w4bf16", "_w5bf16")) for n in on), on


def test_bf16_probe_cells_are_their_fp32_twins_plus_the_flag():
    """The twin relationship the precision arm is read through: a `_w4bf16`
    cell must differ from its `_w4` sibling in `train_autocast_bf16` and in
    NOTHING else, so a bf16-vs-fp32 gap is chargeable to the precision.

    Covers BOTH couplings. At the floor the pair reads as a cost lever; at
    sigma_c it is the quality question, and the twin-ness has to hold at the
    coupling where the columns can actually separate."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    pairs = list(_d400_precision_pairs(CONFIGS))
    assert len(pairs) == 4, [p[0] for p in pairs]
    for label, fp32, bf16 in pairs:
        assert bf16.train.train_autocast_bf16 is True, label
        assert fp32.train.train_autocast_bf16 is False, label
        rebuilt = replace(
            bf16, name=fp32.name,
            train=replace(bf16.train, train_autocast_bf16=False),
        )
        assert rebuilt == fp32, label


def test_frozen_eval_stays_fp32_on_a_bf16_trained_cell():
    """The precision discipline: bf16 is scoped to the loss update, so the
    end-of-run eval config of a bf16-trained cell is indistinguishable from
    its fp32 twin's. A cell trained in bf16 is still evaluated in fp32."""
    from experiments.constrained_hard_03.configs import CONFIGS

    for label, fp32, bf16 in _d400_precision_pairs(CONFIGS):
        assert bf16.eval == fp32.eval, label


def test_training_autocast_is_wired_into_the_swap_trainer():
    """The flag must actually reach the loss update -- a config field no
    code reads is the failure mode this catches, and it would look like a
    null result rather than a bug."""
    import inspect

    from discrete_flow_sampler.samplers import swap_training

    source = inspect.getsource(swap_training.train_swap)
    assert "train_autocast_bf16" in source
