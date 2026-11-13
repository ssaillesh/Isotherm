"""Turn trace jobs into Pod manifests for the kwok cluster."""

from __future__ import annotations

import re

from loader import Job

GPU_RESOURCE = "nvidia.com/gpu"
ANNOTATION_GPU_UTIL = "isotherm.io/gpu-util"
LABEL_RUN = "isotherm.io/replay-run"


def pod_name(job: Job) -> str:
    """DNS-1123 compliant pod name derived from the job id."""
    name = re.sub(r"[^a-z0-9-]+", "-", job.job_id.lower()).strip("-")
    return name[:63].rstrip("-") or "job"


def pod_manifest(job: Job, *, namespace: str, scheduler_name: str, run_id: str) -> dict:
    # Lifecycle (delete after duration_s) is owned by the replayer, so the pod deliberately
    # carries no duration annotation: the simulator must not expire it on its own clock.
    gpus = str(job.gpu_count)
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name(job),
            "namespace": namespace,
            "labels": {LABEL_RUN: run_id},
            "annotations": {ANNOTATION_GPU_UTIL: str(job.gpu_util)},
        },
        "spec": {
            "schedulerName": scheduler_name,
            "restartPolicy": "Never",
            # kwok nodes are fake: select and tolerate them explicitly.
            "nodeSelector": {"type": "kwok"},
            "tolerations": [
                {"key": "kwok.x-k8s.io/node", "operator": "Exists", "effect": "NoSchedule"}
            ],
            "containers": [
                {
                    "name": "work",
                    "image": "registry.k8s.io/pause:3.10",
                    "resources": {
                        "requests": {GPU_RESOURCE: gpus},
                        "limits": {GPU_RESOURCE: gpus},
                    },
                }
            ],
        },
    }
