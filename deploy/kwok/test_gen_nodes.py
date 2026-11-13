import yaml

from gen_nodes import CONFIG_DIR, node_manifests


def test_one_node_per_server_with_config_names():
    layout = yaml.safe_load((CONFIG_DIR / "rack.yaml").read_text())["layout"]
    nodes = node_manifests()
    assert len(nodes) == layout["n_racks"] * layout["slots_per_rack"]
    names = [n["metadata"]["name"] for n in nodes]
    assert len(set(names)) == len(names)
    assert names[0] == layout["node_name_format"].format(rack=0, slot=0)


def test_gpu_capacity_and_kwok_markers():
    gpus = str(yaml.safe_load((CONFIG_DIR / "gpu_specs.yaml").read_text())["node"]["gpus_per_node"])
    for n in node_manifests():
        assert n["status"]["allocatable"]["nvidia.com/gpu"] == gpus
        assert n["metadata"]["labels"]["type"] == "kwok"
        assert n["spec"]["taints"][0]["key"] == "kwok.x-k8s.io/node"
