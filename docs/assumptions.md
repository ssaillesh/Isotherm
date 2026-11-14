# Modelling assumptions

Every number in `configs/` is an assumption. This file records where each one comes from,
how confident we are in it, and what would replace it with a measurement. When you change a
value in `configs/`, update its row here in the same change.

Status key: **sourced** = taken from a primary spec or standard · **derived** = computed from
sourced values · **assumed** = modelling choice with no direct source yet (calibrate later).

Last reviewed: 2026-11-14.

## Layout: `configs/rack.yaml` → `layout`

| Key | Value | Status | Source and notes |
|---|---|---|---|
| `n_racks` | 10 | assumed | A single row of 10 racks is a modelling choice. For scale: NVIDIA's DGX SuperPOD H100 design uses 8 racks of 4 systems per 32-node scalable unit [N3]. |
| `slots_per_rack` | 4 | sourced | NVIDIA's H100 SuperPOD planning guide calls 4 DGX H100 systems per rack the "optimal" density, at about 40.8 kW per rack [N3]. **Caveat:** NVIDIA's FAQ says 2 per rack is the practical maximum in air-cooled facilities unless the room is built for that density [N4]. Our model is air-cooled, so 4 per rack is a deliberately stressed layout. The YAML comment says "8 GPU servers per rack", but the value is 4 slots of 8-GPU servers. Worth rewording to avoid confusion. |
| `gpu_model_by_rack` | racks 0–4 H100, 5–9 A100 | assumed | A placeholder mix that gives the scheduler heterogeneous servers to choose between. No source; edit freely. |
| `node_name_format` | `rack{rack:02d}-slot{slot}` | n/a | A naming convention, not a physical value. Read by `sim/` and `deploy/kwok/` so node names agree without being hardcoded. |

## Cooling: `configs/rack.yaml` → `cooling`

| Key | Value | Status | Source and notes |
|---|---|---|---|
| `supply_temp_c` | 18.0 °C | sourced (as a bound) | The lower edge of the ASHRAE TC 9.9 *recommended* inlet envelope (18–27 °C) [A1]. Real facilities often supply warmer air to save energy, so 18 °C is an optimistic (cool) choice. |
| `inlet_limit_c` | 27.0 °C | sourced | The upper edge of the same ASHRAE recommended envelope [A1]. It is a recommendation, not a hard limit: the *allowable* classes A1–A4 go higher (A1 to 32 °C). |

### Stress scenarios: variant supply temperatures (not in `configs/`)

The stage 2 stress sweep (PLAN.md sessions 4 and 5) runs the sim on copies of `configs/` under
`data/configs/supply<T>/` that differ only in `supply_temp_c` (made with
`make variant-config SUPPLY=<T>`). `configs/` itself stays at 18.0 °C. These temperatures are
stress settings chosen from the v1 results (simulated), not claims about any real facility.

| Variant | Value | Status | Source and notes |
|---|---|---|---|
| `supply24.5` | 24.5 °C | derived | Inlet scenario. 18 + smallest v1 `min_headroom_c` (6.567 °C, spread at load 0.9 seed 0, in `docs/results/v1_supply18/sweep.csv`) = 24.567, rounded **down** to 0.5 °C. The hottest v1 inlet (20.43 °C) would sit at 26.93 °C, just under the 27 °C limit. Inside the ASHRAE recommended envelope [A1]. |
| `supply32` | 32.0 °C | derived | GPU scenario. 18 + smallest v1 gap between a GPU and its own throttle temperature (14.213 °C), rounded **down** to 0.5 °C. The gap was measured by rerunning the hottest v1 run (binpack, load 0.9, seed 4) at 18 °C: the hottest GPU was an H100 at 72.787 °C against its 87 °C throttle (the A100s in that run peaked at 67.59 °C, gap 17.41 °C). See `design.md`, "Stress scenarios". At the top of the A1 *allowable* range [A1], and above the 27 °C inlet limit: every inlet is over the limit in this scenario. |

## Air: `configs/rack.yaml` → `air`

| Key | Value | Status | Source and notes |
|---|---|---|---|
| `airflow_kg_s` | 0.7 kg/s per server | assumed, near spec | The DGX H100 user guide gives 1,105 CFM front-to-back at 80 % fan PWM [N1]. That is 0.522 m³/s, or **≈0.63 kg/s** at ρ = 1.2 kg/m³. The DGX A100 gives 840 CFM, or ≈0.48 kg/s [N5]. So 0.7 kg/s is about 12 % above the H100 rating and about 47 % above the A100 one. For comparison, NVIDIA's design minimum of 157 CFM per kW [N3] works out to about 1,600 CFM (≈0.9 kg/s) at 10.2 kW. |
| `cp_j_kgk` | 1005 J/(kg·K) | sourced | Specific heat of dry air near room temperature. Textbook air tables give 1.007 kJ/(kg·K) at 300 K [T1]. The 0.2 % difference is negligible here. |

Derived check: at full H100 load a node draws 1500 + 8 × 700 = 7,100 W, which gives an
exhaust rise of 7100 / (0.7 × 1005) ≈ **10.1 °C**. At DGX H100's 10.2 kW maximum and its
rated ≈0.63 kg/s, the rise would be ≈16 °C. Our default node is therefore cooler-running than
the vendor's worst case (see `base_w` below).

## Recirculation: `configs/rack.yaml` → `recirculation`

The structure follows the heat-recirculation / cross-interference literature: a share of
each server's exhaust reaches other servers' inlets, falling off with distance, worse near the
top of racks and at the ends of rows [R1, R2]. Those papers fit the coefficients per data
center, from CFD or sensor measurements. **None of the four values below has a published
value we can borrow.** They are placeholders to calibrate.

| Key | Value | Status | Source and notes |
|---|---|---|---|
| `max_fraction` | 0.25 | assumed | The model form comes from [R1, R2]; the magnitude is our choice. To calibrate, fit it against CFD (for example an OpenFOAM or 6SigmaDCX model of this row) or against inlet sensor data. |
| `length_scale` | 1.5 slots | assumed | Exponential distance decay is our modelling choice; [R2] fits a full matrix instead. |
| `height_bias` | 1.0 | assumed | Hot air rising over the rack top is the classic failure mode [R1]. The magnitude is not sourced. |
| `end_bias` | 0.5 | assumed | Wrap-around at row ends is commonly observed [R1]. The magnitude is not sourced. |

Structural assumptions in how `sim/physics/airflow.py` uses these values (full math in its
docstring):

- **One rack pitch equals one slot height** in the distance d_ij. A real rack pitch (≈0.6 m) is larger than a server's height, so this overstates rack-to-rack coupling.
- **Self-recirculation is included.** d_ii = 0, so part of a server's recirculated exhaust lands back on its own inlet.
- **Equal airflow per server.** All servers use `airflow_kg_s`, regardless of load or GPU model. Real fans speed up with load.
- **Air is quasi-static.** Inlet temperatures respond to power instantly (T_in = T_sup + D·P); only GPU junctions carry thermal mass.
- **Feasibility is checked.** Each inlet must take in less recirculated air than its own airflow (column sums of A < 1). With the current values the largest column sum is 0.22.

## GPUs: `configs/gpu_specs.yaml` → `gpus`

| Key | Value | Status | Source and notes |
|---|---|---|---|
| `h100_sxm.tdp_w` | 700 W | sourced | NVIDIA H100 datasheet: SXM max TDP "up to 700 W (configurable)" [N2]. |
| `a100_sxm.tdp_w` | 400 W | sourced | NVIDIA A100 datasheet: 400 W for SXM4 40 GB and 80 GB [N6]. A footnote allows up to 500 W on the HGX A100-80GB custom-thermal SKU, which we ignore. |
| `h100_sxm.r_c_per_w` | 0.075 °C/W | assumed | Junction-to-inlet resistance. Chosen so a 700 W H100 runs about 52 °C above inlet. To calibrate, fit it to steady-state `nvidia-smi` temperature against power at a known inlet temperature. |
| `h100_sxm.c_j_per_c` | 900 J/°C | assumed | Gives τ = R·C ≈ 68 s. To calibrate, fit the step response after a load change. |
| `h100_sxm.throttle_c` | 87 °C | assumed | To calibrate, `nvidia-smi -q -d TEMPERATURE` reports each card's "GPU Slowdown Temp". Use that value. |
| `a100_sxm.r_c_per_w` | 0.12 °C/W | assumed | Same method as for H100. 400 W gives about 48 °C above inlet. |
| `a100_sxm.c_j_per_c` | 650 J/°C | assumed | τ ≈ 78 s. |
| `a100_sxm.throttle_c` | 85 °C | assumed | Read "GPU Slowdown Temp" from `nvidia-smi` as above. |

## Node: `configs/gpu_specs.yaml` → `node`

| Key | Value | Status | Source and notes |
|---|---|---|---|
| `gpus_per_node` | 8 | sourced | DGX H100 and DGX A100 both carry 8 GPUs [N1, N5]. |
| `base_w` | 1500 W | assumed, **likely low** | Comparing vendor maximums: DGX H100 is 10.2 kW max [N1, N4], so 10.2 − 8 × 0.7 ≈ **4.6 kW** non-GPU. DGX A100 is 6.5 kW max [N5], so 6.5 − 8 × 0.4 ≈ **3.3 kW**. Those are PSU-sizing maximums, not typical draw, but 1.5 kW is well under both. To calibrate, measure wall power on an idle-GPU node and subtract GPU idle draw. |
| `idle_gpu_fraction` | 0.10 | assumed, unsourced | We found no primary measurement of H100 SXM idle power. One anecdotal report puts A100 SXM idle near 67 W (≈17 % of TDP) [W1]. That is not authoritative. To calibrate, run `nvidia-smi --query-gpu=power.draw --format=csv -l 1` on idle cards over several minutes. Note that nvidia-smi samples only part of the runtime on A100/H100 and has about ±5 % error [W2]. |

## What these values imply (check after any change)

Computed by the simulator on 2026-11-13, with all 320 GPUs at 100 % utilization held to steady state:

| Quantity | Result | Limit |
|---|---|---|
| Inlet temperature, min–max | 18.7–20.7 °C | 27 °C (`inlet_limit_c`) |
| Hottest exhaust | 30.8 °C | n/a |
| Hottest GPU | 73.2 °C | 85–87 °C (`throttle_c`) |
| Throttled GPUs | 0 | n/a |

**Neither limit binds, even at full load.** Placement still changes margins, so ranking
works, but no placement is ever infeasible. To study a regime where placement decides whether
limits are crossed, the main levers are a warmer `supply_temp_c`, a larger `max_fraction`, a
lower `airflow_kg_s`, or a `base_w` closer to the vendor figures above. Which one to move is a
modelling decision. The hottest inlets sit at the top slot of rack 0 (an end rack), as
`height_bias` and `end_bias` intend.

## References

- [A1] ASHRAE TC 9.9, *Thermal Guidelines for Data Processing Environments*, 5th ed., 2021. Recommended envelope 18–27 °C dry bulb. Secondary summary: [Sunbird DCIM](https://sunbirddcim.com/blog/does-your-data-center-follow-ashrae-tc-99-thermal-guidelines-data-processing-environments). Cite the ASHRAE book in papers.
- [N1] NVIDIA, *DGX H100/H200 System User Guide*, "Introduction": 10.2 kW max, 1,105 CFM at 80 % PWM, 8 GPUs. <https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html>
- [N2] NVIDIA, *H100 Tensor Core GPU datasheet*: SXM max TDP up to 700 W. <https://www.nvidia.com/en-us/data-center/h100/>
- [N3] NVIDIA, *DGX SuperPOD Data Center Design (DGX H100)*, "Planning": 4 systems per rack optimal, ≈40.8 kW per rack, 157 CFM per kW. <https://docs.nvidia.com/dgx-superpod/design-guides/dgx-superpod-data-center-design-h100/latest/planning.html>
- [N4] NVIDIA, *DGX SuperPOD FAQ*: 2 per rack practical maximum when air-cooled. <https://docs.nvidia.com/dgx-superpod/faq/latest/dgx-superpod.html>
- [N5] NVIDIA, *DGX A100 System User Guide*, "Introduction": 6.5 kW max, 840 CFM at 80 % PWM. <https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html>
- [N6] NVIDIA, *A100 Tensor Core GPU datasheet*: SXM 400 W. <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>
- [T1] Incropera, DeWitt, Bergman, Lavine, *Fundamentals of Heat and Mass Transfer*, Table A.4 (air at 1 atm), cp = 1.007 kJ/(kg·K) at 300 K.
- [R1] J. Moore, J. Chase, P. Ranganathan, R. Sharma, "Making Scheduling 'Cool': Temperature-Aware Workload Placement in Data Centers," USENIX ATC 2005.
- [R2] Q. Tang, S. K. S. Gupta, G. Varsamopoulos, "Energy-Efficient Thermal-Aware Task Scheduling for Homogeneous High-Performance Computing Data Centers: A Cyber-Physical Approach," *IEEE TPDS* 19(11), 2008.
- [W1] Anecdotal blog report of A100 idle draw (not authoritative): <https://dev.to/mikebains41debug/how-i-discovered-a-hidden-146w-power-draw-on-nvidia-a100-gpus-and-built-an-open-source-fix-1n8h>
- [W2] "Part-time Power Measurements: nvidia-smi's Lack of Attention," arXiv:2312.02741. <https://arxiv.org/abs/2312.02741>
