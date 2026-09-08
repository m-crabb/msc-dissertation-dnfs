"""torch.compile opt-in for the flip-route rate model.

Mirrors the hard route's compile_head contract (test_swap_perf_refactors
::test_compile_head_flag_matches_uncompiled_and_keeps_state_dict):
`ModelCfg.compile_model=True` must use in-place nn.Module.compile so
state_dict keys stay unprefixed (torch.compile(module) wrapping would add
`_orig_mod.` and break checkpoint round-trips) and the compiled forward
agrees with eager at fp32 tolerance. The soft route reuses this ModelCfg
and `run.train`, so the flag serves both chapters. Default False = every
archived cell byte-identical.
"""

from types import SimpleNamespace

import torch
from experiments.dnfs_baseline_01.configs import ModelCfg
from experiments.dnfs_baseline_01.run import _build_model

from discrete_flow_sampler.targets.ising import IsingTarget


def test_compile_model_default_off():
    assert ModelCfg.__dataclass_fields__["compile_model"].default is False


@torch.no_grad()
def test_compile_model_flag_matches_uncompiled_and_keeps_state_dict():
    target = IsingTarget(D=2, sigma=0.1)
    plain_cfg = SimpleNamespace(model=ModelCfg(kind="lemlp", hidden_dim=16, n_layers=2))

    torch.manual_seed(0)
    plain_model = _build_model(plain_cfg, target)
    keys_before = sorted(plain_model.state_dict())
    torch.manual_seed(1)
    x = target.sample_base(4, device="cpu")
    t = torch.rand(4)
    want = plain_model(x, t)

    compiled_cfg = SimpleNamespace(
        model=ModelCfg(kind="lemlp", hidden_dim=16, n_layers=2, compile_model=True)
    )
    torch.manual_seed(0)
    compiled_model = _build_model(compiled_cfg, target)
    assert sorted(compiled_model.state_dict()) == keys_before
    got = compiled_model(x, t)
    assert torch.allclose(got, want, atol=1e-5)


def test_optimised_recipe_flips_only_the_declared_flags():
    """The flip-route `optimised_recipe` is the Wave-1 landing vehicle:
    exactly model.compile_model and train.c_t_from_rollout flip, every
    other field is untouched (twin discipline — the transform must never
    smuggle a third change into a new cell)."""
    from dataclasses import fields

    from experiments.dnfs_baseline_01.configs import CONFIGS, optimised_recipe

    base = CONFIGS["stage_4_d10"]
    assert base.model.compile_model is False
    assert getattr(base.train, "c_t_from_rollout", False) is False
    optimised = optimised_recipe(base)
    assert optimised.model.compile_model is True
    assert optimised.train.c_t_from_rollout is True
    for field in fields(base):
        if field.name in ("model", "train"):
            continue
        assert getattr(optimised, field.name) == getattr(base, field.name)
    for field in fields(base.model):
        if field.name == "compile_model":
            continue
        assert getattr(optimised.model, field.name) == getattr(base.model, field.name)
    for field in fields(base.train):
        if field.name == "c_t_from_rollout":
            continue
        assert getattr(optimised.train, field.name) == getattr(base.train, field.name)
