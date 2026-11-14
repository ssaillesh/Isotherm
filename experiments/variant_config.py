"""Write a variant config directory that differs from configs/ only in supply air temperature.

Sweeps never edit configs/ (CLAUDE.md rule 2). They point the sim at a copy instead:

    uv run python variant_config.py --supply 24.5 --out ../data/configs/supply24.5

gpu_specs.yaml is copied byte for byte. In rack.yaml only the value on the
`supply_temp_c:` line changes; every other byte, comments included, stays the same. This is
a text edit, not a YAML round trip, so the copy diffs against configs/ in exactly one line.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SUPPLY_LINE = re.compile(r"^(\s*supply_temp_c:\s*)([-+0-9.eE]+)", re.MULTILINE)


def with_supply(rack_text: str, supply_c: float) -> str:
    """rack.yaml text with the supply_temp_c value replaced; ValueError unless one match."""
    matches = SUPPLY_LINE.findall(rack_text)
    if len(matches) != 1:
        raise ValueError(f"expected one supply_temp_c line in rack.yaml, found {len(matches)}")
    return SUPPLY_LINE.sub(lambda m: f"{m.group(1)}{float(supply_c)!r}", rack_text)


def read_supply(config_dir: Path) -> float:
    """The supply_temp_c value in <config_dir>/rack.yaml (same one-line rule as with_supply)."""
    matches = SUPPLY_LINE.findall((config_dir / "rack.yaml").read_text())
    if len(matches) != 1:
        raise ValueError(f"expected one supply_temp_c line in {config_dir}/rack.yaml")
    return float(matches[0][1])


def write_variant(base: Path, out: Path, supply_c: float) -> Path:
    """Create `out` as a copy of `base` with a new supply temperature.

    An existing `out` is accepted only if it already holds exactly that content, so a rerun
    is harmless but a variant directory is never silently overwritten.
    """
    files = {
        "rack.yaml": with_supply((base / "rack.yaml").read_text(), supply_c).encode(),
        "gpu_specs.yaml": (base / "gpu_specs.yaml").read_bytes(),
    }
    for name, content in files.items():
        if (out / name).exists() and (out / name).read_bytes() != content:
            raise ValueError(f"{out / name} exists with other content; remove it or pick --out")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_bytes(content)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base", type=Path, default=Path("../configs"))
    p.add_argument("--supply", type=float, required=True, help="supply air temperature, °C")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        out = write_variant(args.base, args.out, args.supply)
    except ValueError as e:
        p.error(str(e))
    print(f"wrote {out} ({args.base} with supply_temp_c: {args.supply!r})")


if __name__ == "__main__":
    main()
