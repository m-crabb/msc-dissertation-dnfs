"""Score the 64-site Cu-Au cells without a reference chain.

At 64 sites the fixed-composition slice cannot be enumerated (C(64,32) ~ 1.8e18); this
script reads what the sampler's own artefacts can certify:

  ESS            raw and EMA eval ESS (5000 draws), the headline;
  E/site (meV)   mean, min and IS-weighted (self-normalised log weights) sample energy against
                 the ground state of the slice: L1_0 at c=0.5 (-36.23 meV/site), L1_2 at c=0.25
                 (-32.88); kT at 500 K is 43 meV for the whole cell, so an ordered sampler sits within
                 ~1 meV/site of it and every wrong swap costs ~7 kT;
  swaps          unlike-pair swaps from each sample to its nearest ordered variant (all translates
                 and axis choices), i.e. half the Hamming distance; the uniform slice sits ~13.4
                 (c50) so ~0 means ordered, ~10 means multi-domain;
  loss/static    end-of-stage training loss over the static-flow (identity) loss of that stage,
                 Var_uniform[beta_stage E] estimated on 4096 uniform slice states: ~0 = the flow
                 moves the slice, ~1 = it has given up (c50 patch cells: 0.12 -> 1.0);
  E_ref          the reference chain's energy per site at the cell's final temperature when
                 results/alloy_ref/cuau64_chain_<c25|c50|free>_T<T>.json exists (reference_chain.py).

Free-ensemble cells (A1_*) have no slice: E_ground, swaps and the static variance use the
uniform free ensemble at the sampled composition, and the sample composition is printed.

Usage:  pixi run -e dev python -m experiments.alloy_ce.judge_64site_cells "results/03_hard/*cuau64* results/02_constrained_soft/*cuau64*"
"""

import glob
import json
import os
import re
import sys

import pandas as pd
import torch
from experiments.alloy_ce.tools.patch_reach_probe import (
    ordered_states,
    random_slice_states,
)

from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

K_B = 8.617333262e-5
MEV = 1000.0


def swaps_to_nearest_ordered(samples, refs):
    """Half the minimum Hamming distance to any ordered variant: one unlike swap flips two sites."""
    hamming = (samples[:, None, :] != refs[None, :, :]).sum(-1)  # (n_samples, n_refs)
    return hamming.min(1).values / 2.0


def stage_loss_over_static(run, df, spec, beta_of_sigma, n_sites, n_plus):
    """End-of-stage mean loss / Var_uniform[beta E] for every curriculum stage."""
    cfg = json.load(open(run + "/config.json"))
    stages = cfg["curriculum"]["stages"]
    n_steps = cfg["train"]["n_steps"]
    ends = [s["start_step"] for s in stages[1:]] + [n_steps]
    slice_states = random_slice_states(
        4096, n_sites, n_plus, torch.Generator().manual_seed(0)
    ).double()
    energies = spec.energy(slice_states)
    ratios = []
    for stage, end in zip(stages, ends):
        beta = beta_of_sigma(stage["sigma"])
        window = df[(df.step >= end - 500) & (df.step < end)]["loss"].mean()
        ratios.append(window / (beta * energies).var())
    return ratios


def reference_energy(c_tag, T_final):
    path = f"results/alloy_ref/cuau64_chain_{c_tag}_T{T_final}.json"
    if not os.path.exists(path):
        return float("nan")
    return json.load(open(path))["energy_per_site"]["mean"] * MEV


def main(pattern):
    rows = []
    for run in sorted(p for g in pattern.split() for p in glob.glob(g)):
        if not glob.glob(run + "/eval/metrics.json"):
            continue
        cfg = json.load(open(run + "/config.json"))
        spec = BinaryExpansionSpec.from_json(cfg["ising"]["expansion_json"])
        n_sites = len(spec.positions)
        c = cfg["ising"].get("target_composition")
        free = run.split("/")[-1].startswith("A1_")
        beta_of_sigma = lambda sigma: 2.0 * sigma  # sigma = beta/2 on the alloy cells
        beta_final = beta_of_sigma(cfg["curriculum"]["stages"][-1]["sigma"])
        T_final = round(1.0 / (K_B * beta_final))
        if free:
            samples = torch.load(run + "/eval/samples.pt").double()
            c = ((samples + 1) / 2).mean().item()  # the sampled composition
            refs = ordered_states(
                spec, "l10"
            ).double()  # nearest ordered = a formality here
        else:
            refs = ordered_states(
                spec, "l10" if abs(c - 0.5) < 1e-6 else "l12"
            ).double()
        n_plus = round(c * n_sites)
        e_ground = spec.energy(refs).min().item() / n_sites * MEV
        df = pd.read_csv(run + "/training_log.csv")
        ratios = stage_loss_over_static(run, df, spec, beta_of_sigma, n_sites, n_plus)
        c_tag = "free" if free else f"c{round(100 * c)}"
        row = {
            "cell": re.sub(r"_20\d{6}-[a-z0-9-]+$", "", run.split("/")[-1]),
            "T": T_final,
            "c_sample": round(c, 3),
            "E_ref": reference_energy(c_tag, T_final),
        }
        for sub in ("eval", "eval_ema"):
            m = json.load(open(f"{run}/{sub}/metrics.json"))
            row[f"ess_{sub}"] = m["ess_fraction"]
            samples = torch.load(f"{run}/{sub}/samples.pt").double()
            log_w = torch.load(f"{run}/{sub}/log_weights.pt").double()
            e_site = spec.energy(samples) / n_sites * MEV
            w = torch.softmax(log_w, 0)
            swaps = swaps_to_nearest_ordered(samples, refs)
            row.update(
                {
                    f"E_mean_{sub}": e_site.mean().item(),
                    f"E_min_{sub}": e_site.min().item(),
                    f"E_is_{sub}": (w * e_site).sum().item(),
                    f"swaps_mean_{sub}": swaps.mean().item(),
                    f"swaps_is_{sub}": (w * swaps).sum().item(),
                    f"std_logw_{sub}": log_w.std().item(),
                }
            )
        row["E_ground"] = e_ground
        row["loss/static per stage"] = " ".join(f"{r:.2f}" for r in ratios)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("cell")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(
        table[
            [
                "T",
                "c_sample",
                "E_ref",
                "E_ground",
                "ess_eval",
                "ess_eval_ema",
                "std_logw_eval",
            ]
        ]
        .round(3)
        .to_string()
    )
    print()
    print(
        table[
            [
                "E_mean_eval",
                "E_min_eval",
                "E_is_eval",
                "swaps_mean_eval",
                "swaps_is_eval",
            ]
        ]
        .round(2)
        .to_string()
    )
    print()
    print(
        table[
            [
                "E_mean_eval_ema",
                "E_is_eval_ema",
                "swaps_mean_eval_ema",
                "swaps_is_eval_ema",
            ]
        ]
        .round(2)
        .to_string()
    )
    print()
    print(table[["loss/static per stage"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/03_hard/*cuau64*")
