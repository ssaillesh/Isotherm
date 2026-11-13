"""Load rack layout and GPU specs from configs/ (the single source of truth)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# In the repo, configs/ sits next to sim/. In the container it's copied to /app/configs
# and ISOTHERM_CONFIG_DIR points at it.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def config_dir() -> Path:
    return Path(os.environ.get("ISOTHERM_CONFIG_DIR", DEFAULT_CONFIG_DIR))


@dataclass(frozen=True)
class GpuSpec:
    model: str
    tdp_w: float
    r_c_per_w: float
    c_j_per_c: float
    throttle_c: float


@dataclass(frozen=True)
class NodeSpec:
    name: str
    rack: int  # 0 .. n_racks-1 along the row
    slot: int  # 0 = bottom
    gpu: GpuSpec


@dataclass(frozen=True)
class Recirculation:
    max_fraction: float
    length_scale: float
    height_bias: float
    end_bias: float


@dataclass(frozen=True)
class ClusterSpec:
    n_racks: int
    slots_per_rack: int
    supply_temp_c: float
    inlet_limit_c: float
    airflow_kg_s: float
    cp_j_kgk: float
    recirculation: Recirculation
    gpus_per_node: int
    base_w: float
    idle_gpu_fraction: float
    nodes: tuple[NodeSpec, ...]  # rack-major: rack 0 slots 0..S-1, then rack 1, ...


def load_cluster(rack_path: Path | None = None, gpu_specs_path: Path | None = None) -> ClusterSpec:
    rack_path = rack_path or config_dir() / "rack.yaml"
    gpu_specs_path = gpu_specs_path or config_dir() / "gpu_specs.yaml"
    rack = yaml.safe_load(rack_path.read_text())
    gpu_cfg = yaml.safe_load(gpu_specs_path.read_text())

    gpus = {
        model: GpuSpec(model=model, **{k: float(v) for k, v in p.items()})
        for model, p in gpu_cfg["gpus"].items()
    }
    layout = rack["layout"]
    n_racks, slots = int(layout["n_racks"]), int(layout["slots_per_rack"])
    models = layout["gpu_model_by_rack"]
    if len(models) != n_racks:
        raise ValueError(f"gpu_model_by_rack has {len(models)} entries, n_racks is {n_racks}")
    for m in models:
        if m not in gpus:
            raise ValueError(f"gpu_model_by_rack uses unknown model {m!r}; known: {sorted(gpus)}")
    fmt = layout["node_name_format"]
    nodes = tuple(
        NodeSpec(name=fmt.format(rack=r, slot=s), rack=r, slot=s, gpu=gpus[models[r]])
        for r in range(n_racks)
        for s in range(slots)
    )
    if len({n.name for n in nodes}) != len(nodes):
        raise ValueError(f"node_name_format {fmt!r} produces duplicate names")

    node = gpu_cfg["node"]
    return ClusterSpec(
        n_racks=n_racks,
        slots_per_rack=slots,
        supply_temp_c=float(rack["cooling"]["supply_temp_c"]),
        inlet_limit_c=float(rack["cooling"]["inlet_limit_c"]),
        airflow_kg_s=float(rack["air"]["airflow_kg_s"]),
        cp_j_kgk=float(rack["air"]["cp_j_kgk"]),
        recirculation=Recirculation(**{k: float(v) for k, v in rack["recirculation"].items()}),
        gpus_per_node=int(node["gpus_per_node"]),
        base_w=float(node["base_w"]),
        idle_gpu_fraction=float(node["idle_gpu_fraction"]),
        nodes=nodes,
    )
