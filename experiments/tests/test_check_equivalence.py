import pytest

from check_equivalence import compare_rows, read_rows, speed_rows


def _rows(**cols):
    base = {"policy": "binpack", "target_load": "0.3", "seed": "0", "peak_temp_c": "72.5"}
    row = {**base, **cols}
    return {(row["policy"], row["target_load"], row["seed"]): row}


def test_identical_rows_have_no_differences():
    assert compare_rows(_rows(), _rows()) == []


def test_last_digit_difference_is_reported():
    [line] = compare_rows(_rows(peak_temp_c="72.50000000000001"), _rows())
    assert "peak_temp_c" in line and "72.50000000000001" in line


def test_missing_rows_and_columns_are_reported():
    other = _rows(policy="spread")
    problems = compare_rows(_rows(), other, ("old", "new"))
    assert problems == [
        "('binpack', '0.3', '0'): only in old",
        "('spread', '0.3', '0'): only in new",
    ]
    assert compare_rows(_rows(extra="1"), _rows()) == [
        "('binpack', '0.3', '0') extra: 1 (A) != None (B)"
    ]


def test_read_rows_keys_by_policy_load_seed(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("policy,target_load,seed,peak_temp_c\nrandom,0.9,0,71.0\n")
    assert read_rows(path) == _rows(policy="random", target_load="0.9", peak_temp_c="71.0")


def test_speed_rows_total_per_load_and_overall():
    a = {("x", "0.3", "0"): 10.0, ("y", "0.3", "0"): 30.0, ("x", "0.9", "0"): 8.0}
    b = {("x", "0.3", "0"): 5.0, ("y", "0.3", "0"): 10.0, ("x", "0.9", "0"): 4.0}
    rows = speed_rows(a, b)
    assert rows[:3] == [("x", "0.3", 10.0, 5.0), ("y", "0.3", 30.0, 10.0), ("x", "0.9", 8.0, 4.0)]
    assert rows[3:] == [
        ("all", "0.3", 40.0, 15.0),
        ("all", "0.9", 8.0, 4.0),
        ("all", "all", pytest.approx(48.0), pytest.approx(19.0)),
    ]
