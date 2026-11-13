# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Emit fake kwok Node manifests for every server in configs/rack.yaml.

Node names, rack/slot labels, GPU model and GPU count all come from configs/, so the kwok
cluster always matches what the simulator models.

    uv run deploy/kwok/gen_nodes.py | kubectl apply -f -
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(
    os.environ.get("ISOTHERM_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs")
)

# Fake CPU/memory capacity so pods never block on them. Not part of the thermal model;
# replayed pods only request GPUs.
FAKE_CPU, FAKE_MEMORY, FAKE_PODS = "224", "2Ti", "110"


def node_manifests(config_dir: Path = CONFIG_DIR) -> list[dict]:
    rack = yaml.safe_load((config_dir / "rack.yaml").read_text())["layout"]
    gpus_per_node = str(
        yaml.safe_load((config_dir / "gpu_specs.yaml").read_text())["node"]["gpus_per_node"]
    )
    out = []
    for r in range(int(rack["n_racks"])):
        for s in range(int(rack["slots_per_rack"])):
            name = rack["node_name_format"].format(rack=r, slot=s)
            resources = {
                "cpu": FAKE_CPU,
                "memory": FAKE_MEMORY,
                "pods": FAKE_PODS,
                "nvidia.com/gpu": gpus_per_node,
            }
            out.append(
                {
                    "apiVersion": "v1",
                    "kind": "Node",
                    "metadata": {
                        "name": name,
                        "annotations": {
                            "node.alpha.kubernetes.io/ttl": "0",
                            "kwok.x-k8s.io/node": "fake",
                        },
                        "labels": {
                            "type": "kwok",
                            "kubernetes.io/hostname": name,
                            "kubernetes.io/role": "agent",
                            "isotherm.io/rack": str(r),
                            "isotherm.io/slot": str(s),
                            "isotherm.io/gpu-model": rack["gpu_model_by_rack"][r],
                        },
                    },
                    "spec": {
                        "taints": [
                            {"key": "kwok.x-k8s.io/node", "value": "fake", "effect": "NoSchedule"}
                        ]
                    },
                    "status": {"allocatable": resources, "capacity": dict(resources)},
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    args = p.parse_args()
    yaml.safe_dump_all(node_manifests(args.config_dir), sys.stdout, sort_keys=False)


if __name__ == "__main__":
    main()
