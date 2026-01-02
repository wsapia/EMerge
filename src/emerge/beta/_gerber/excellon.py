# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

import re
from typing import List, Dict, Optional, Tuple

class ExcellonParseError(Exception):
    pass

def parse_excellon(
    text: str,
    *,
    convert_to: Optional[str] = "mm",   # "mm", "inch", or None for native
) -> List[Dict[str, float]]:
    """
    Parse an Excellon (NC drill) file string and return a list of drill hits with radii.

    Returns: List of dicts with keys:
      - x, y (floats, in 'convert_to' units if provided, else native file units)
      - radius (float, same units)
      - unit ("mm" or "inch")
      - tool (e.g., "T01")

    Features:
      • Supports INCH/METRIC (also M72/M71).
      • Supports zero suppression LZ/TZ.
      • Uses ;FILE_FORMAT=<int>:<dec> if present to interpret implicit decimals.
      • Modal coordinates: missing X or Y reuse last value.
      • Tool table with diameters via lines like: T1F00S00C0.03543
      • Heuristic for “short fields”: If a coordinate has fewer digits than expected,
        it tries both LZ- and TZ-style padding and chooses the value closer to the
        previous axis value (helps with occasional writer quirks).
      • Ignores non-drill routing (basic: lines without X/Y are ignored unless R repeats).
    """
    # Defaults (updated by headers)
    unit = None            # "inch" or "mm"
    zero_sup = "LZ"        # "LZ" or "TZ"
    int_digits, dec_digits = 2, 4  # sensible default if FILE_FORMAT missing
    in_header = False

    # Tooling
    tool_diam_in_native_units: dict[str, float] = {}  # e.g. {"T1": 0.8}
    current_tool: Optional[str] = None

    # Modal state
    last_x: Optional[float] = None
    last_y: Optional[float] = None

    # Results
    hits: List[Dict[str, float]] = []

    # Regex helpers
    re_filefmt = re.compile(r";\s*FILE_FORMAT\s*=\s*(\d+)\s*:\s*(\d+)", re.I)
    re_units_line = re.compile(r"^\s*(INCH|METRIC)\s*(?:,?\s*(LZ|TZ))?\s*$", re.I)
    re_tool_def = re.compile(r"^\s*T(\d+)[^;\r\n]*?C\s*=?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",re.I)
    re_tool_sel = re.compile(r"^\s*T(\d+)\s*$", re.I)
    re_M71 = re.compile(r"^\s*M71\b")  # metric
    re_M72 = re.compile(r"^\s*M72\b")  # inch

    # Coordinate line patterns (allow just X, just Y, or both, and optional R repeat)
    re_xy = re.compile(
        r"^(?:.*?)(?:X(-?\d+))?(?:Y(-?\d+))?(?:R(\d+))?\s*$",
        re.I
    )

    def to_native_units(val: float) -> float:
        return val  # already native; conversion applied later in one go

    def convert(val: float, from_unit: str, to_unit: Optional[str]) -> float:
        if to_unit is None or to_unit == from_unit:
            return val
        if from_unit == "inch" and to_unit == "mm":
            return val * 25.4
        if from_unit == "mm" and to_unit == "inch":
            return val / 25.4
        raise ExcellonParseError(f"Unknown conversion {from_unit}→{to_unit}")

    def parse_number_field(
        s: str,
        last_val: Optional[float],
        int_d: int,
        dec_d: int,
        zero_rule: str
    ) -> float:
        """
        Parse a digit string without a decimal point using the given format.
        Applies LZ/TZ rule, but also tries a heuristic "other side" padding
        and picks the value closer to the last value on that axis if available.
        """
        # If the writer included a decimal point (rare in coords), just float it
        if "." in s or s.startswith(("+", "-")) and "." in s[1:]:
            try:
                return float(s)
            except ValueError:
                raise ExcellonParseError(f"Bad numeric field: {s}")

        total = int_d + dec_d
        raw = s.strip()
        neg = raw.startswith("-")
        if neg:
            raw = raw[1:]
        if not raw.isdigit():
            raise ExcellonParseError(f"Bad digits in coordinate: {s}")

        def to_val(padded: str) -> float:
            # split from the RIGHT by dec_d (more robust if length != expected)
            if dec_d == 0:
                integ, frac = padded, ""
            else:
                frac = padded[-dec_d:] if len(padded) >= dec_d else padded.rjust(dec_d, "0")
                integ = padded[:-dec_d] if len(padded) >= dec_d else ""
            v = (int(integ) if integ else 0) + (int(frac) / (10 ** dec_d) if dec_d > 0 else 0.0)
            return -v if neg else v

        # Primary padding based on declared suppression
        if zero_rule.upper() == "LZ":
            primary = raw.rjust(total, "0")
            alt = raw.ljust(total, "0")
        else:  # TZ
            primary = raw.ljust(total, "0")
            alt = raw.rjust(total, "0")

        v_primary = to_val(primary)
        if last_val is None:
            # no history—trust header rule
            return v_primary

        v_alt = to_val(alt)

        # Choose whichever is closer to the previous modal value on this axis.
        # Require a meaningful improvement to avoid flip-flopping on small noise.
        d_p = abs(v_primary - last_val)
        d_a = abs(v_alt - last_val)

        if d_a + 1e-12 < d_p * 0.2:  # alt is at least 5x closer → take alt
            return v_alt
        return v_primary

    # Pass 1: parse header + body
    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Skip blank lines & pure comments
        if not line or line.startswith(";"):
            # However, parse FILE_FORMAT even if comment-style
            mfmt = re_filefmt.search(raw_line)
            if mfmt:
                int_digits, dec_digits = int(mfmt.group(1)), int(mfmt.group(2))
            continue

        if line.upper().startswith("M48"):
            in_header = True
            continue
        if line == "%":
            # end of header or section; continue
            in_header = False
            continue

        # Units (either in header line or body)
        m_units = re_units_line.match(line)
        if m_units:
            unit = m_units.group(1).lower()
            if m_units.group(2):
                zero_sup = m_units.group(2).upper()
            continue

        if re_M71.match(line):
            unit = "mm"
            continue
        if re_M72.match(line):
            unit = "inch"
            continue

        # Tool definition lines (e.g., T1F00S00C0.03543)
        m_tool_def = re_tool_def.match(line)
        if m_tool_def:
            tnum = int(m_tool_def.group(1))
            tool = f"T{tnum:02d}"
            diam = float(m_tool_def.group(2))
            tool_diam_in_native_units[tool] = diam
            continue

        # Tool selection (e.g., T01)
        m_tool_sel = re_tool_sel.match(line)
        if m_tool_sel:
            tnum = int(m_tool_sel.group(1))
            current_tool = f"T{tnum:02d}"
            # Some writers chain selection with coords on same line, but our regex below catches coords too.
            continue

        # Stop codes etc.
        up = line.upper()
        if up.startswith("M30") or up.startswith("M00") or up.startswith("M95"):
            break

        # Coordinate line: may contain X, Y, optional R
        m_xy = re_xy.match(line)
        if not m_xy:
            # Unknown line; ignore quietly (robust)
            continue

        sx, sy, sr = m_xy.group(1), m_xy.group(2), m_xy.group(3)
        if sx is None and sy is None:
            # nothing here
            continue

        # Parse X/Y with modal reuse if missing
        if sx is not None:
            x_native = parse_number_field(sx, last_x, int_digits, dec_digits, zero_sup)
            last_x = to_native_units(x_native)
        if sy is not None:
            y_native = parse_number_field(sy, last_y, int_digits, dec_digits, zero_sup)
            last_y = to_native_units(y_native)

        if last_x is None or last_y is None:
            # need both to place a hole
            continue

        repeat = int(sr) if sr else 1

        # Determine radius from current tool
        if current_tool and current_tool in tool_diam_in_native_units:
            diam_native = tool_diam_in_native_units[current_tool]
        else:
            # Unknown tool diameter → leave radius as None
            diam_native = None

        # Output in desired units
        final_unit = unit or "mm"  # default to inch if totally unspecified (rare)
        for _ in range(repeat):
            if diam_native is None:
                radius_out = None
            else:
                radius_out = convert(diam_native / 2.0, final_unit, convert_to)
            hits.append({
                "x": convert(last_x, final_unit, convert_to),
                "y": convert(last_y, final_unit, convert_to),
                "radius": radius_out,
                "unit": convert_to or final_unit,
                "tool": current_tool or "T??",
            })

    return hits


def parse_excellon_file(path: str, **kwargs) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_excellon(f.read(), **kwargs)
