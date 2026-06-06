# Appendix: Experiment Hyperparameters

This appendix table summarizes the settings used for the paper experiments. In the final paper terminology, **ST-LoRA** denotes stage-wise transferability-aware LoRA, and **STPlanner** is its planner. In older result files and scripts, the same planner may appear as `STaR-v2`.

## Global Training and Evaluation Settings

| Category | Hyperparameter | Value used in paper experiments | Notes |
|---|---:|---|---|
| Repeated runs | Seeds | 42, 43, 44 | Reported table values are mean +/- std over these three seeds unless otherwise noted. |
| Explicit seed protocol | Split/model/residual/LoRA/head/loader seeds | Set explicitly from the run seed | For SPECTRA/BRE runs, residual/BRE initialization also receives the same explicit seed. |
| Data split | Sen1Floods11 | Fixed split files `sen1floods11_s{seed}_{train,val,test}.txt` | Split seed follows the run seed for Sen1Floods11. |
| Data split | FireScars | Fixed split files `fire_scars_{train,val,test}.txt` | This corresponds to split seed 42. |
| Data split | Landslide4Sense | Fixed split files `landslide4sense_{train,val,test}.txt` | This corresponds to split seed 42. |
| Data split | GEO-Bench SA Crop Type | Official `default_partition.json` train/valid/test split | No random split is used for GEO-Bench. |
| Input normalization | All datasets | Per-band 2nd-98th percentile normalization to `[0, 1]` | Applied independently per image band. |
| Train augmentation | All datasets | Random crop + random horizontal flip | Validation/test use center crop, except fixed-size chips that only need padding. |
| Batch size | All datasets/backbones | 8 | One GPU per training process. |
| DataLoader workers | All datasets/backbones | 4 | Loader RNG is controlled by `loader_seed`. |
| Optimizer | All methods | AdamW | No LR scheduler; fixed group learning rates are used. |
| Weight decay | Default | 0.05 | Patch-embedding tuning diagnostics used 0.0 for the patch-embedding group, but Table 1 methods keep the patch embedding frozen. |
| Gradient clipping | Default | Global norm 1.0 | Applied after backpropagation before optimizer step. |
| Checkpoint selection | Validation | Best checkpoint selected by validation mIoU | Test metrics are reported from held-out best-checkpoint evaluation. |
| Test visualization | Automatic evaluation | Best/final checkpoint test visualizations saved when `--auto-test` is enabled | Binary overlays use task-specific foreground/error coloring; multi-class overlays use a fixed palette. |

## Dataset and Backbone-Specific Settings

| Backbone | Dataset | Epochs | Crop / image size | Patch-size multiple | Decoder | Feature layers connected to UPerNet | Decoder channels |
|---|---|---:|---:|---:|---|---|---:|
| Prithvi-EO-2.0 600M | Sen1Floods11 | 50 | 224 | 14 | UPerNet | `[7, 15, 23, 31]` | 256 |
| Prithvi-EO-2.0 600M | FireScars | 10 | 224 | 14 | UPerNet | `[7, 15, 23, 31]` | 256 |
| Prithvi-EO-2.0 600M | Landslide4Sense | 50 | 128 | 14 | UPerNet | `[7, 15, 23, 31]` | 256 |
| Prithvi-EO-2.0 600M | GEO-Bench SA Crop Type | 50 | 224 | 14 | UPerNet | `[7, 15, 23, 31]` | 256 |
| ScaleMAE | Sen1Floods11 | 50 | 224 | 16 | UPerNet | `[7, 11, 15, 23]` | 256 |
| ScaleMAE | FireScars | 10 | 224 | 16 | UPerNet | `[7, 11, 15, 23]` | 256 |
| ScaleMAE | Landslide4Sense | 50 | 128 | 16 | UPerNet | `[7, 11, 15, 23]` | 256 |
| ScaleMAE | GEO-Bench SA Crop Type | 50 | 224 | 16 | UPerNet | `[7, 11, 15, 23]` | 256 |
| SatMAE | Sen1Floods11 | 50 | 128 | 8 | UPerNet | `[5, 11, 17, 23]` | 256 |
| SatMAE | FireScars | 10 | 128 | 8 | UPerNet | `[5, 11, 17, 23]` | 256 |
| SatMAE | Landslide4Sense | 50 | 128 | 8 | UPerNet | `[5, 11, 17, 23]` | 256 |
| SatMAE | GEO-Bench SA Crop Type | 50 | 96 | 8 | UPerNet | `[5, 11, 17, 23]` | 256 |

## Learning Rates

| Parameter group | Paper value | Applies to | Notes |
|---|---:|---|---|
| LoRA adapters | `5e-5` | LoRA-8/16/32/64 and ST-LoRA | This is the half-LR setting used for the final matched runs. |
| Decoder/head | `5e-4` | All fine-tuning methods | UPerNet decoder and 1x1 segmentation head are optimized together as the task head. |
| Unfrozen backbone weights | `5e-6` | Full-FT, last-stage, surgical FT | 10x lower than LoRA LR to reduce destructive updates to pretrained features. |
| BRE / residual embedding adapter | `5e-4` unless overridden | LoRA+BRE and SPECTRA | The residual/BRE adapter uses the residual group LR, which falls back to the head LR in the current implementation. |
| SSC-PE fine-tuning group | `1e-4` in configs; not part of final Table 1 SPECTRA rows unless SSC-PE is active | Historical cross-sensor adapter path | The paper SPECTRA rows use BRE/D or native embedding rather than SSC-PE as the main embedding component. |
| Patch embedding | Frozen | Table 1 methods | Patch-embedding-tunable runs were diagnostic ablations, not main Table 1 settings. |

## Loss and Class Balancing

| Component | Hyperparameter | Value | Notes |
|---|---:|---|---|
| Main paper loss | Loss mode | `ce_dice_dwa` for most final runs | Dynamic Weight Average over CE/weighted CE and Dice. |
| Fixed CE+Dice variant | Loss mode | `ce_dice`, `dice_lambda = 1.0` | Used for selected runs where the run ID contains `ce_dice1`; e.g., SatMAE + Sen1Floods11 SPECTRA. |
| DWA temperature | `T` | 2.0 | First two epochs use equal weights; from epoch 3, `w_i = 2 * softmax((L_i[t-1] / L_i[t-2]) / T)`. |
| CE term | Class weights | `class_weights: auto` | CE is weighted when auto weights are active. |
| Binary class weighting | Scheme | Inverse-frequency + minority boost | Minority class weight is multiplied by 2.0 when the backbone is adapting. |
| Binary boost cap | Cap ratio | 8.0 in final paper runs | Minority weight is capped at `8.0 x` majority weight. |
| Multi-class class weighting | GEO-Bench SA Crop Type | ENet/log-inverse weighting, `log_smoothing = 1.02` | Weights are normalized by their mean over present classes. |
| Absent classes | Weight | 0.0 | Prevents unbounded inverse-frequency weights for absent classes. |
| Dice implementation | Probability input | Softmax probabilities | Dice is computed from soft probabilities, not argmax predictions. |
| Dice ignore mask | Ignore index | `-1` | Ignored pixels are excluded from CE and Dice. |
| Dice foreground handling | Binary datasets | Foreground-only; class 0 excluded | `include_background = false`, `class_mode = foreground/default`. |
| Dice foreground handling | GEO-Bench SA Crop Type | Present train classes, background/no-data excluded | `class_mode = present_train`, `include_background = false`. |
| Empty valid batches | Dice behavior | Returns zero loss contribution | Prevents NaNs when no valid pixels/classes are present. |

## ST-LoRA / STPlanner Hyperparameters

| Hyperparameter | Value | Meaning |
|---|---:|---|
| Planner names in paper | STPlanner inside ST-LoRA | Older result files may call this `STaR-v2`. |
| Planner modes | `transfer`, `repair` | `transfer` allocates more rank to high-LogME stages; `repair` allocates more rank to high-gap stages. |
| Number of stages | 4 | ViT blocks are partitioned into four stage groups. |
| Backbone freezing | Frozen backbone, LoRA-only | STPlanner returns `unfrozen = [False, False, False, False]`; only LoRA, decoder/head, and embedding adapter are trained. |
| Reference uniform rank | 32 | Auto budget is defined relative to uniform LoRA-32, i.e., max total rank budget `32 x 4 = 128`. |
| Rank grid | `[4, 8, 16, 32, 64]` | Per-stage rank is snapped to this discrete grid. |
| Minimum per-stage rank | 4 | No stage receives rank 0 in STPlanner-v2 experiments. |
| Candidate total rank budgets | `[32, 48, 60, 72, 80, 92, 96, 104, 112]` | The continuous auto budget is snapped down to this set, capped by the reference budget. |
| Budget fraction range | `f_min = 0.40`, `f_max = 0.85` | Auto-selected budget lies between 40% and 85% of the uniform LoRA-32 budget. |
| Budget mapping | `f(q) = f_min + (f_max - f_min) * 0.5 * (1 - tanh(slope * (q_norm - midpoint)))` | Lower normalized transferability receives a larger LoRA budget. |
| Budget midpoint | 0.50 | Center of the tanh budget mapping. |
| Budget slope | 3.0 | Controls sharpness of the transferability-to-budget mapping. |
| Softmax temperature | `tau = 0.05` | Used when converting stage transfer/gap scores into rank allocation weights. |
| Stage prior | `[0.8, 1.0, 1.1, 1.2]` | Mildly favors deeper stages before softmax normalization. |
| Transferability score | `q_s` | Stagewise LogME score computed from frozen/backbone features on target training data. |
| Overall transferability | `q_overall = max_s q_s` | Used to determine total rank budget. |
| Normalization source | `results/spectral_mismatch_transferability.csv` | Provides the q-bank min/max used to normalize `q_overall`. |
| Repair gap | `delta_q_s = q_max - q_s` | Used by the `repair` strategy to allocate rank to lower-transferability stages. |

## Selected SPECTRA Planner Choices in Table 1

| Backbone | Dataset | Embedding component | Planner / rank schedule used in Table 1 |
|---|---|---|---|
| Prithvi-EO-2.0 600M | Sen1Floods11 | BRE-PoG | STPlanner repair; ranks varied by seed, e.g. `[8,32,8,8]`, `[8,16,16,8]`, `[4,16,8,8]` |
| Prithvi-EO-2.0 600M | FireScars | Native embedding | Manual stage-wise schedule `[32,16,16,32]`; FireScars already matches the 6-band Prithvi input, so BRE is bypassed. |
| Prithvi-EO-2.0 600M | Landslide4Sense | D residual | Legacy MGAS/STaR-v1 schedule `[4,4,4,4]` |
| Prithvi-EO-2.0 600M | GEO-Bench SA Crop Type | BRE-PoG | STPlanner repair `[16,32,16,32]` |
| ScaleMAE | Sen1Floods11 | BRE-PoG | STPlanner transfer `[16,32,16,16]` |
| ScaleMAE | FireScars | BRE-PoG | Manual stage-wise schedule `[8,16,32,32]` |
| ScaleMAE | Landslide4Sense | BRE-PoG | STPlanner transfer; ranks varied by seed, e.g. `[8,16,16,8]`, `[4,8,32,4]`, `[8,16,16,4]` |
| ScaleMAE | GEO-Bench SA Crop Type | BRE-PoG | STPlanner repair; ranks varied by seed, e.g. `[16,16,16,32]`, `[32,16,16,32]` |
| SatMAE | Sen1Floods11 | BRE-PoG | STPlanner transfer `[8,16,16,16]`; CE+Dice1 loss variant |
| SatMAE | FireScars | BRE-PoG | Manual stage-wise schedule `[16,32,32,16]` |
| SatMAE | Landslide4Sense | BRE-PoG | STPlanner transfer; ranks varied by seed, e.g. `[8,8,16,16]`, `[4,16,16,16]`, `[8,16,16,16]` |
| SatMAE | GEO-Bench SA Crop Type | BRE-PoG | STPlanner repair; ranks varied by seed, e.g. `[32,32,16,16]`, `[16,32,16,32]` |

