import difflib
from pathlib import Path

import pytest

from variant_config import read_supply, with_supply, write_variant

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_variant_differs_from_configs_in_one_line_only(tmp_path):
    out = write_variant(REPO_CONFIGS, tmp_path / "supply24.5", 24.5)
    base = (REPO_CONFIGS / "rack.yaml").read_text().splitlines()
    variant = (out / "rack.yaml").read_text().splitlines()
    changed = [d for d in difflib.ndiff(base, variant) if d[:1] in "+-"]
    assert len(changed) == 2  # one line out, one line in
    assert changed[1].lstrip("+ ").startswith("supply_temp_c: 24.5")
    assert (out / "gpu_specs.yaml").read_bytes() == (REPO_CONFIGS / "gpu_specs.yaml").read_bytes()


def test_supply_value_and_comment_are_kept_apart():
    text = "cooling:\n  supply_temp_c: 18.0      # cold air\n  inlet_limit_c: 27.0\n"
    assert with_supply(text, 24.5) == (
        "cooling:\n  supply_temp_c: 24.5      # cold air\n  inlet_limit_c: 27.0\n"
    )


@pytest.mark.parametrize(
    "text", ["cooling:\n  inlet_limit_c: 27.0\n", "supply_temp_c: 18\nsupply_temp_c: 19\n"]
)
def test_needs_exactly_one_supply_line(text):
    with pytest.raises(ValueError, match="expected one"):
        with_supply(text, 24.5)


def test_rerun_is_harmless_but_never_overwrites(tmp_path):
    out = tmp_path / "v"
    write_variant(REPO_CONFIGS, out, 24.5)
    write_variant(REPO_CONFIGS, out, 24.5)  # same content: fine
    with pytest.raises(ValueError, match="other content"):
        write_variant(REPO_CONFIGS, out, 30.0)
    assert "supply_temp_c: 24.5" in (out / "rack.yaml").read_text()


def test_supply_is_written_exactly_not_rounded():
    assert "supply_temp_c: 24.25" in with_supply("supply_temp_c: 18.0\n", 24.25)


def test_read_supply_from_a_variant(tmp_path):
    assert read_supply(write_variant(REPO_CONFIGS, tmp_path / "v", 24.5)) == 24.5
