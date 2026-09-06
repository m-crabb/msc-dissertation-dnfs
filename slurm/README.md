# Cluster launcher index

[Project overview](../README.md) · [Experiment guide](../experiments/README.md)

This index covers all 97 retained shell launchers in this directory. They record
specific campaigns, including unsuccessful, cancelled and superseded experiments.
The seven Cu–Au launch/pull wrappers are indexed separately in the
[alloy tools guide](../experiments/alloy_ce/tools/README.md#historical-launch-and-transfer-wrappers).

## Execution context

The Slurm scripts contain Imperial DoC paths, GPU requests, environment setup,
config selections, tags and sometimes warm-start parents. Inspect the selected
file and its Python entrypoint before adapting it to a new environment. A shell
exit status or existing metrics file does not certify complete artifacts;
some historical loops continue after child failures. Many gate thresholds are
recorded in comments rather than enforced by the wrapper.

Campaign decisions remain part of the record:

- All `mars_*` campaigns are permanently withdrawn; do not submit them.
- `d256_rung_clip50.sbatch` was retired before launch; do not submit it.
- The `fine`/ne512 arm of `d256_fmo2_warm.sbatch` was cancelled; do not relaunch it.
- `hard_camort_d256_seeds.sbatch` is a protected historical snapshot. It predates
  the per-slice training-baseline correction and must remain unchanged.
- Preserve each campaign's frozen gates, selected seeds/checkpoints and timing
  exclusions. MPS or mixed-venue timings are not interchangeable with the
  controlled benchmark results.

The groups below describe purpose, not current scheduler state or permission to
rerun. Fixed-tag retries, side evaluations and pull wrappers can encounter
existing archives; keep original outputs and verify each entrypoint's contract.

## Environment and smoke checks

| Launcher | Purpose / retained status |
| --- | --- |
| [env_probe.sbatch](env_probe.sbatch) | CUDA/library and import probe |
| [train_smoke.sbatch](train_smoke.sbatch) | Small hard-training integration run |
| [train_smoke_offline.sbatch](train_smoke_offline.sbatch) | Small hard-training run with offline W&B |
| [d256_smoke.sbatch](d256_smoke.sbatch) | Four-step d256 integration run |

## Baseline, soft constraints and free energy

| Launcher | Purpose / retained status |
| --- | --- |
| [wave1_unconstrained_sc.sbatch](wave1_unconstrained_sc.sbatch) | Exact-critical unconstrained wave |
| [walkback_d8_baseline.sbatch](walkback_d8_baseline.sbatch) | Baseline walkback campaign |
| [walkback_d8_soft.sbatch](walkback_d8_soft.sbatch) | Soft walkback campaign |
| [topup_walkback.sh](topup_walkback.sh) | Historical walkback top-up |
| [amort_d4_validate.sbatch](amort_d4_validate.sbatch) | 4×4 soft amortisation validation |
| [amort_d10.sbatch](amort_d10.sbatch) | 10×10 soft amortisation |
| [softcamort_d8.sbatch](softcamort_d8.sbatch) | 8×8 soft composition amortisation |
| [obedience_probe.sbatch](obedience_probe.sbatch) | Soft-penalty obedience probe |
| [efc_lambda_sweep.sbatch](efc_lambda_sweep.sbatch) | Exact-field-channel penalty sweep |
| [d10_sc_hardrecipe_pair.sbatch](d10_sc_hardrecipe_pair.sbatch) | 10×10 hard-recipe flip controls |
| [d16_unconstrained_control.sbatch](d16_unconstrained_control.sbatch) | 16×16 unconstrained control |
| [fc_d10_ne128.sbatch](fc_d10_ne128.sbatch) | 10×10 free-energy runs |
| [fc_c030_topup.sbatch](fc_c030_topup.sbatch) | Composition-0.30 free-energy top-up |
| [fc_redraw_grids.sbatch](fc_redraw_grids.sbatch) | Stored-model evaluations on finer grids |

## Hard architecture gates and smaller lattices

| Launcher | Purpose / retained status |
| --- | --- |
| [factorised_gate_4x4.sbatch](factorised_gate_4x4.sbatch) | Factorised-head exact-system gate |
| [oracle_gate_4x4.sbatch](oracle_gate_4x4.sbatch) | Doubly-hollow oracle comparison |
| [interior_gate_4x4.sbatch](interior_gate_4x4.sbatch) | Interior-head arm gate |
| [interior_d64_rung.sbatch](interior_d64_rung.sbatch) | 8×8 interior-head rung |
| [ctv_d64_naive_twin.sbatch](ctv_d64_naive_twin.sbatch) | 8×8 naive-estimator twin |
| [fac_d64_rung.sbatch](fac_d64_rung.sbatch) | 8×8 row-factorised head |
| [fab16_d64_rung.sbatch](fab16_d64_rung.sbatch) | 8×8 rank-16 factorised head |
| [fmo2_d64_rung.sbatch](fmo2_d64_rung.sbatch) | 8×8 row/column-factorised head |
| [d64_fmo2_h128.sbatch](d64_fmo2_h128.sbatch) | 8×8 factorised width comparison |
| [d144_fmo2_rung.sbatch](d144_fmo2_rung.sbatch) | 12×12 factorised rung |
| [d144_ma_bracket.sbatch](d144_ma_bracket.sbatch) | 12×12 masked-attention bracket |
| [m2_d64_ctema4_gate.sbatch](m2_d64_ctema4_gate.sbatch) | Training-baseline EMA experiment |
| [m3_d64_ctb512_smoke.sbatch](m3_d64_ctb512_smoke.sbatch) | 8×8 training-baseline batch experiment |
| [m3_d256_ctb512_smoke.sbatch](m3_d256_ctb512_smoke.sbatch) | 16×16 training-baseline batch experiment |
| [m6_d64_replay2_smoke.sbatch](m6_d64_replay2_smoke.sbatch) | 8×8 replay-window experiment |
| [hard_d64_floor.sbatch](hard_d64_floor.sbatch) | 8×8 subcritical comparison |
| [wave2_d64.sbatch](wave2_d64.sbatch) | 8×8 hard house-table wave |

## Historical d256 screens, continuations and transfer

| Launcher | Purpose / retained status |
| --- | --- |
| [d256_rung.sbatch](d256_rung.sbatch) | Original masked-attention curriculum |
| [d256_rung_naive.sbatch](d256_rung_naive.sbatch) | Naive-estimator rescue curriculum |
| [d256_smoke12k_arm.sbatch](d256_smoke12k_arm.sbatch) | Short curriculum arm, optional warm start |
| [d256_scr5k_mo.sbatch](d256_scr5k_mo.sbatch) | Mask-one screen |
| [d256_screen_arm.sbatch](d256_screen_arm.sbatch) | Config-selected screen wrapper |
| [d256_cv2.sbatch](d256_cv2.sbatch) | Control-variate continuation of rescue |
| [d256_clip2000_cont.sbatch](d256_clip2000_cont.sbatch) | Clip-2000 continuation of rescue |
| [d256_fmo2_warm.sbatch](d256_fmo2_warm.sbatch) | Coarse warm transfer retained; fine/ne512 arm CANCELLED, do not relaunch |
| [d256_cv_cont_recipe.sbatch](d256_cv_cont_recipe.sbatch) | Recipe continuation with control variate |
| [d256_naive_cont_recipe.sbatch](d256_naive_cont_recipe.sbatch) | Matched naive-estimator continuation |
| [d256-A-cvcont-ne128.sbatch](d256-A-cvcont-ne128.sbatch) | A: ne128 control-variate continuation |
| [d256-B-cv70k-ne128.sbatch](d256-B-cv70k-ne128.sbatch) | B: ne128 control-variate run |
| [d256-Bef-cv70k-ne128.sbatch](d256-Bef-cv70k-ne128.sbatch) | Bef: exact-field twin of B |
| [d256-C-cvcont-h128L3-fromP.sbatch](d256-C-cvcont-h128L3-fromP.sbatch) | C: wider continuation from P |
| [d256-C-cvcont-h128L3-fromP03.sbatch](d256-C-cvcont-h128L3-fromP03.sbatch) | C: wider continuation from P03 |
| [d256-D-h128L3-cv70k.sbatch](d256-D-h128L3-cv70k.sbatch) | D: wider control-variate run |
| [d256-Dlr03-h128L3-cv70k.sbatch](d256-Dlr03-h128L3-cv70k.sbatch) | D: learning-rate variant |
| [d256-P-h128L3-naive.sbatch](d256-P-h128L3-naive.sbatch) | P: wider naive-estimator parent |
| [d256-P03-h128L3-lr03.sbatch](d256-P03-h128L3-lr03.sbatch) | P03: learning-rate parent variant |
| [d256_house_ma_s010.sbatch](d256_house_ma_s010.sbatch) | Masked-attention subcritical house arm; retained run halted at 5k of configured 50k |

## Hard scaling, raster and composition campaigns

| Launcher | Purpose / retained status |
| --- | --- |
| [d400_thp2_bf16.sbatch](d400_thp2_bf16.sbatch) | 20×20 subcritical, radius 2, bf16 |
| [d400_thp2_fp32.sbatch](d400_thp2_fp32.sbatch) | 20×20 subcritical, radius 2, fp32 |
| [d400_thp3_bf16.sbatch](d400_thp3_bf16.sbatch) | 20×20 subcritical, radius 3, bf16 |
| [d400_thp3_fp32.sbatch](d400_thp3_fp32.sbatch) | 20×20 subcritical, radius 3, fp32 |
| [d576_thp3_bf16.sbatch](d576_thp3_bf16.sbatch) | 24×24 critical, radius 3, bf16 |
| [d576_thp4_bf16.sbatch](d576_thp4_bf16.sbatch) | 24×24 critical, radius 4, bf16 |
| [rasterord_d64_floor.sbatch](rasterord_d64_floor.sbatch) | 8×8 raster-order floor |
| [rasterord_d256_floor.sbatch](rasterord_d256_floor.sbatch) | 16×16 raster-order floor |
| [rasterord_d256_a100_resume.sbatch](rasterord_d256_a100_resume.sbatch) | 16×16 raster-order continuation |
| [hard_camort_d256_perslice.sbatch](hard_camort_d256_perslice.sbatch) | Composition-amortised per-slice campaign |
| [hard_camort_d256_seeds.sbatch](hard_camort_d256_seeds.sbatch) | Protected historical additional-seed snapshot; predates per-slice correction |

## Evaluation, references and transport

| Launcher | Purpose / retained status |
| --- | --- |
| [hard_probe_replicates.sbatch](hard_probe_replicates.sbatch) | Independent evaluation-seed probes |
| [smc_tau_sweep.sbatch](smc_tau_sweep.sbatch) | SMC threshold evaluation sweep |
| [m5_d256_smc_eval.sbatch](m5_d256_smc_eval.sbatch) | SMC evaluations for selected rescue runs |
| [rw_stage_best_eval.sbatch](rw_stage_best_eval.sbatch) | Selected-stage checkpoint evaluation |
| [n_euler_sweep_d256.sbatch](n_euler_sweep_d256.sbatch) | Euler-grid resolution sweep |
| [d256_keystone_grid_probe.sbatch](d256_keystone_grid_probe.sbatch) | EMA grid-comparison evaluations |
| [a2_transport_backfill.sbatch](a2_transport_backfill.sbatch) | Transport diagnostic backfill |
| [d256-N5-transport.sbatch](d256-N5-transport.sbatch) | Continuation transport decomposition |
| [d256_zero_shot_transfer.sbatch](d256_zero_shot_transfer.sbatch) | Zero-shot coupling transfer |
| [mchammer_baselines.sbatch](mchammer_baselines.sbatch) | Canonical and VC-SGC reference generation |

## Neural comparators and compute measurements

| Launcher | Purpose / retained status |
| --- | --- |
| [mdns_budget_gate2_4x4.sbatch](mdns_budget_gate2_4x4.sbatch) | Budget-masked diffusion second gate |
| [mdns_budget_gate3_4x4.sbatch](mdns_budget_gate3_4x4.sbatch) | Budget-masked diffusion third gate |
| [m4a_mdns_budget_8x8.sbatch](m4a_mdns_budget_8x8.sbatch) | 8×8 budget-masked diffusion comparison |
| [gfn_d64_wave.sbatch](gfn_d64_wave.sbatch) | 8×8 GFlowNet wave |
| [gfn_d256_wave.sbatch](gfn_d256_wave.sbatch) | 16×16 GFlowNet wave |
| [gfn_d400_wave.sbatch](gfn_d400_wave.sbatch) | 20×20 GFlowNet wave |
| [retry_gfn_d256.sh](retry_gfn_d256.sh) | Historical GFlowNet retry wrapper |
| [gfn_d64_launch_bench.sbatch](gfn_d64_launch_bench.sbatch) | 8×8 GFlowNet benchmark launch |
| [gfn_d256_launch_bench.sbatch](gfn_d256_launch_bench.sbatch) | 16×16 GFlowNet benchmark launch |
| [bench_eval_wallclock_d64.sbatch](bench_eval_wallclock_d64.sbatch) | 8×8 evaluation wall-clock benchmark |
| [bench_thp_compile.sbatch](bench_thp_compile.sbatch) | Two-hole patch compilation benchmark |
| [bench_thp_venue_control.sbatch](bench_thp_venue_control.sbatch) | Two-hole patch venue control |
| [fmo2_bench_d256.sbatch](fmo2_bench_d256.sbatch) | 16×16 factorised-head benchmark |

## Withdrawn campaigns — do not submit

| Launcher | Purpose / retained status |
| --- | --- |
| [d256_rung_clip50.sbatch](d256_rung_clip50.sbatch) | Retired before launch on 11 August 2026 |
| [mars_d256_recipe_mps.sbatch](mars_d256_recipe_mps.sbatch) | MARS permanently withdrawn on 31 August 2026 |
| [mars_d256_two_cfg_mps.sbatch](mars_d256_two_cfg_mps.sbatch) | MARS permanently withdrawn on 31 August 2026 |
| [mars_d400_mps.sbatch](mars_d400_mps.sbatch) | MARS permanently withdrawn on 31 August 2026 |
| [mars_mal_d64_sequential.sbatch](mars_mal_d64_sequential.sbatch) | MARS permanently withdrawn on 31 August 2026 |
| [mars_mar_d64_sequential.sbatch](mars_mar_d64_sequential.sbatch) | MARS permanently withdrawn on 31 August 2026 |
| [mars_marope_d64_sequential.sbatch](mars_marope_d64_sequential.sbatch) | MARS permanently withdrawn on 31 August 2026 |
| [mars_marope2_d64_sequential.sbatch](mars_marope2_d64_sequential.sbatch) | MARS permanently withdrawn on 31 August 2026 |

## Modal interfaces

Remote entrypoints also live in [baseline](../experiments/dnfs_baseline_01/modal_app.py),
[soft](../experiments/constrained_soft_02/modal_app.py),
[hard](../experiments/constrained_hard_03/modal_app.py) and
[configuration calibration](../scripts/modal_configuration_calibration_4x4.py).
These modules declare cloud resources on import. Read their decorated interfaces
and selection defaults before invocation; this index does not verify live cloud
availability or stored volume contents.
