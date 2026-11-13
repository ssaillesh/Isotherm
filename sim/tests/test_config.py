import pytest
import yaml

from config import DEFAULT_CONFIG_DIR, load_cluster


def test_repo_configs_load(spec):
    assert len(spec.nodes) == spec.n_racks * spec.slots_per_rack
    assert len({n.name for n in spec.nodes}) == len(spec.nodes)
    assert spec.supply_temp_c < spec.inlet_limit_c
    for n in spec.nodes:
        assert 0 <= n.slot < spec.slots_per_rack
        assert 0 <= n.rack < spec.n_racks


def test_node_names_come_from_config(spec):
    fmt = yaml.safe_load((DEFAULT_CONFIG_DIR / "rack.yaml").read_text())["layout"][
        "node_name_format"
    ]
    assert spec.nodes[0].name == fmt.format(rack=0, slot=0)


def _write_configs(tmp_path, **layout_overrides):
    rack = yaml.safe_load((DEFAULT_CONFIG_DIR / "rack.yaml").read_text())
    rack["layout"].update(layout_overrides)
    (tmp_path / "rack.yaml").write_text(yaml.safe_dump(rack))
    return tmp_path / "rack.yaml"


def test_gpu_model_list_must_match_racks(tmp_path):
    path = _write_configs(tmp_path, gpu_model_by_rack=["h100_sxm"])
    with pytest.raises(ValueError, match="entries"):
        load_cluster(rack_path=path)


def test_unknown_gpu_model_rejected(tmp_path):
    path = _write_configs(tmp_path, n_racks=1, gpu_model_by_rack=["nope"])
    with pytest.raises(ValueError, match="unknown model"):
        load_cluster(rack_path=path)


def test_name_format_must_be_unique(tmp_path):
    path = _write_configs(tmp_path, node_name_format="rack{rack}")
    with pytest.raises(ValueError, match="duplicate"):
        load_cluster(rack_path=path)
