import csv
import math

import pytest

from compare import compare, format_table, read_sweep, write_compare

HEADER = "policy,target_load,seed,offered_load,peak_temp_c,min_headroom_c,achieved_utilisation"
# Hand-made: isotherm vs binpack at load 0.5, seeds 0-2.
#   peak_temp_c:     d = 70-71, 71-71, 72-74 = -1, 0, -2   mean -1,   std 1
#   min_headroom_c:  d = 5-3,   6-4,   7-4   =  2, 2, 3    mean 7/3,  std sqrt(1/3)
#   utilisation:     d = 0, 0, 0                            mean 0,    std 0
ROWS = """\
isotherm,0.5,0,0.48,70.0,5.0,0.5
isotherm,0.5,1,0.51,71.0,6.0,0.5
isotherm,0.5,2,0.50,72.0,7.0,0.5
binpack,0.5,0,0.48,71.0,3.0,0.5
binpack,0.5,1,0.51,71.0,4.0,0.5
binpack,0.5,2,0.50,74.0,4.0,0.5
"""


@pytest.fixture
def sweep(tmp_path):
    path = tmp_path / "sweep.csv"
    path.write_text(f"{HEADER}\n{ROWS}")
    return path


def _by_metric(results):
    return {r.metric: r for r in results}


def test_trace_columns_are_not_compared(sweep):
    _, metrics = read_sweep(sweep)
    assert metrics == ["peak_temp_c", "min_headroom_c", "achieved_utilisation"]


def test_hand_computed_paired_differences(sweep):
    got = _by_metric(compare(*read_sweep(sweep), "isotherm"))
    peak, head, util = got["peak_temp_c"], got["min_headroom_c"], got["achieved_utilisation"]

    assert (peak.target_load, peak.policy, peak.other, peak.n) == (0.5, "isotherm", "binpack", 3)
    assert (peak.mean_diff, peak.std_diff) == (-1.0, 1.0)
    assert (peak.lower, peak.better, peak.verdict) == (2, 2, "no difference")  # |mean| = std

    assert head.mean_diff == pytest.approx(7 / 3)
    assert head.std_diff == pytest.approx(math.sqrt(1 / 3))
    assert (head.lower, head.better, head.verdict) == (0, 3, "difference")  # higher is better

    assert (util.mean_diff, util.std_diff) == (0.0, 0.0)
    assert (util.lower, util.better, util.verdict) == (0, None, "no difference")


def test_direction_flips_with_the_chosen_policy(sweep):
    got = _by_metric(compare(*read_sweep(sweep), "binpack"))
    assert got["peak_temp_c"].mean_diff == 1.0
    assert (got["peak_temp_c"].lower, got["peak_temp_c"].better) == (0, 0)
    assert (got["min_headroom_c"].lower, got["min_headroom_c"].better) == (3, 0)


def test_seed_sets_must_match(tmp_path):
    path = tmp_path / "sweep.csv"
    path.write_text(f"{HEADER}\n{ROWS}isotherm,0.5,3,0.5,70.0,5.0,0.5\n")
    with pytest.raises(ValueError, match="seeds"):
        compare(*read_sweep(path), "isotherm")


def test_duplicate_rows_and_unknown_policy_are_errors(tmp_path, sweep):
    dup = tmp_path / "dup.csv"
    dup.write_text(f"{HEADER}\n{ROWS}binpack,0.5,2,0.50,74.0,4.0,0.5\n")
    with pytest.raises(ValueError, match="two rows"):
        compare(*read_sweep(dup), "isotherm")
    with pytest.raises(ValueError, match="no rows"):
        compare(*read_sweep(sweep), "ortools")


def test_csv_leaves_better_blank_without_a_direction(sweep, tmp_path):
    out = tmp_path / "compare.csv"
    write_compare(compare(*read_sweep(sweep), "isotherm"), out)
    with out.open(newline="") as f:
        rows = {r["metric"]: r for r in csv.DictReader(f)}
    assert rows["achieved_utilisation"]["better"] == ""
    assert rows["min_headroom_c"]["better"] == "3"
    assert rows["peak_temp_c"]["verdict"] == "no difference"


def test_table_shows_mean_std_count_and_label(sweep):
    results = compare(*read_sweep(sweep), "isotherm")
    table = format_table(results, "peak_temp_c")
    assert "better (lower)" in table
    assert "binpack  -1.00 ± 1.00 (2/3)  no diff" in table
    assert "+2.33 ± 0.58 (3/3)" in format_table(results, "min_headroom_c")


def test_table_adds_a_digit_when_a_real_gap_rounds_to_mean_equals_std(tmp_path):
    # d = 0.01, 0.05, 0.10: mean 0.0533 > std 0.0451, both "0.05" at 2 decimals.
    path = tmp_path / "sweep.csv"
    path.write_text(
        "policy,target_load,seed,peak_inlet_c\n"
        "isotherm,0.9,0,20.01\nisotherm,0.9,1,20.05\nisotherm,0.9,2,20.10\n"
        "binpack,0.9,0,20.00\nbinpack,0.9,1,20.00\nbinpack,0.9,2,20.00\n"
    )
    results = compare(*read_sweep(path), "isotherm")
    assert results[0].verdict == "difference"
    assert "+0.053 ± 0.045 (0/3)" in format_table(results, "peak_inlet_c")
