from dataclasses import fields, replace

from linearity_check import CHECKED, shift_report
from sweep import SweepRow


def _row(**kw) -> SweepRow:
    base = {f.name: 0 for f in fields(SweepRow)}
    base.update(policy="binpack", target_load=0.5, seed=0, peak_inlet_c=20.0, peak_temp_c=70.0)
    base.update(kw)
    return SweepRow(**base)


def test_exact_shift_passes_and_reports_deltas():
    a = _row()
    b = replace(a, peak_inlet_c=26.5, peak_temp_c=76.5)
    passed, rows = shift_report(a, b, 6.5)
    assert passed
    delta = {name: d for name, _, _, d in rows}
    assert delta["peak_inlet_c"] == 6.5 and delta["peak_temp_c"] == 6.5
    assert "policy" not in delta  # key columns are not compared


def test_within_tolerance_passes_and_each_checked_column_counts():
    a = _row()
    assert shift_report(a, replace(a, peak_inlet_c=26.55, peak_temp_c=76.45), 6.5, 0.0501)[0]
    for column in CHECKED:
        off = replace(a, peak_inlet_c=26.5, peak_temp_c=76.5)
        off = replace(off, **{column: getattr(off, column) + 0.06})
        assert not shift_report(a, off, 6.5)[0]
