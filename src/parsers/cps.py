"""Bronze-layer parsing of raw CPS Basic / CPS Mare-Winship extractor files
into a tidy DataFrame, persisted as bronze parquet.

Reads the as-saved zip (fixed-width data file) from extractors.cps plus its
covering SPS dictionary, and produces one row per person-record with one
typed column per SPS variable. Kept separate from extraction so it's
testable on small in-memory fixed-width strings without hitting the network.

SPS dictionary format (NBER's CPS files use SPSS syntax):
 - one variable per line
 - `name start-end` with variables ending at a lone "." line, e.g.:

    input program.
    data list file='c:\\cpsmw64.raw' /
                hhcount    1-6
                hhidnum    15
                vnmajid    371-374     (a)
    .
    variable labels
                hhcount    "Household Counter"
                ...
    .
    value labels
                famdesc
                       1   "primary fam containing no subfams"
                       2   "primary fam with 1 or more subfams"
                /famtypc
                       1   "yes"
                       2   "no"
                .

- A trailing `(a)` marks alphanumeric (string) variable
- Everything else is numeric
- `variable labels` block contains variable description
- `value labels` block contains values descriptions
- SPS-files are transfromed into dicts of the following format:
    {
        "start": int,
        "end": int,
        "numeric": bool,
        "Description": ...,
        "Values": {
            code: label
        }
    }
"""

import io
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.input_output.parquet import write_parquet
from src.schemas.bronze.cps_long import validate_cps_long

_VAR_LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+
    (?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?
    (?:\s*\(\s*(?P<fmt>[Aa])\s*\)\s*)?
    \s*$
    """,
    re.VERBOSE | re.MULTILINE,
)
_VARIABLE_LABEL_BLOCK_RE = re.compile(
    r"^\s*variable labels\s*$(?P<body>.*?)^\s*\.\s*$",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_VARIABLE_LABEL_LINE_RE = re.compile(
    r'^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+"(?P<description>[^"]*)"\s*$'
)
_VALUE_LABEL_BLOCK_RE = re.compile(
    r"^\s*value labels\s*$(?P<body>.*?)^\s*\.\s*$",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_VALUE_LABEL_VAR_HEADER_RE = re.compile(
    r"^\s*/?\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$"
)
_VALUE_LABEL_CODE_RE = re.compile(r'^\s*(?P<code>\d+)\s+"(?P<label>[^"]*)"\s*$')


@dataclass(frozen=True)
class CPSVariable:
    name: str
    start: int  # 1-indexed, inclusive
    end: int  # 1-indexed, inclusive
    numeric: bool


def parse_sps_dictionary(sps_text: str) -> list[CPSVariable]:
    """Parse an SPS `data list file=... /` dictionary into column specs.

    Only the variable-definition lines (between the `/` and the terminating
    lone "." line) are used; the `variable labels` block and everything else
    is ignored.
    """
    match = re.search(
        r"data list.*?/(.*?)^\s*\.\s*$",
        sps_text,
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    body = match.group(1) if match else sps_text

    variables = []
    for m in _VAR_LINE_RE.finditer(body):
        start = int(m.group("start"))
        end = int(m.group("end")) if m.group("end") else start
        variables.append(
            CPSVariable(
                name=m.group("name"),
                start=start,
                end=end,
                numeric=m.group("fmt") is None,
            )
        )
    if not variables:
        raise ValueError("No variable definitions found in SPS dictionary")
    return variables


def load_sps_dictionary(sps_path: Path) -> list[CPSVariable]:
    return parse_sps_dictionary(sps_path.read_text(encoding="latin-1"))


def parse_variable_labels(sps_text: str) -> dict[str, str]:
    """Parse an SPS `variable labels` block into {variable_name: description}.

    One variable per line (no `/` grouping needed, unlike `value labels` —
    there's only ever one description per variable): `name "description"`,
    terminated by a lone "." line, e.g.:

        variable labels
             hhcount    "Household Counter"
             famhh1     "Family-in-Household1"
             .

    Returns {} if the SPS text has no `variable labels` block.
    """
    match = _VARIABLE_LABEL_BLOCK_RE.search(sps_text)
    if not match:
        return {}

    descriptions: dict[str, str] = {}
    for line in match.group("body").splitlines():
        line_match = _VARIABLE_LABEL_LINE_RE.match(line)
        if line_match:
            descriptions[line_match.group("name")] = line_match.group("description")
    return descriptions


def parse_value_labels(sps_text: str) -> dict[str, dict[str, str]]:
    """Parse an SPS `value labels` block into {variable_name: {code: label}}.

    The block groups one variable's numeric-code -> label pairs per `/`
    (first variable has no leading `/`), terminated by a lone "." line, e.g.:

        value labels
             famdesc
                    1     "primary fam containing no subfams"
                    2     "primary fam with 1 or more subfams"
             /famtypc
                    1     "yes"
                    2     "no"
             .

    Codes are kept as strings (not int) since that's both what JSON object
    keys require and what apply_value_labels needs to match against
    stringified DataFrame values. Returns {} if the SPS text has no `value
    labels` block (not every dictionary variant defines one).
    """
    match = _VALUE_LABEL_BLOCK_RE.search(sps_text)
    if not match:
        return {}

    value_labels: dict[str, dict[str, str]] = {}
    current_var: str | None = None
    for line in match.group("body").splitlines():
        header_match = _VALUE_LABEL_VAR_HEADER_RE.match(line)
        if header_match:
            current_var = header_match.group("name")
            value_labels[current_var] = {}  # type: ignore
            continue
        code_match = _VALUE_LABEL_CODE_RE.match(line)
        if code_match and current_var is not None:
            value_labels[current_var][code_match.group("code")] = code_match.group(
                "label"
            )
    return value_labels


def build_variable_dictionary(sps_text: str) -> dict[str, dict[str, object]]:
    """Combine parse_sps_dictionary + parse_variable_labels + parse_value_labels
    into the JSON shape persisted by save_variable_dictionary:

        {variable_name: {"start": int, "end": int, "numeric": bool,
                          "Description": str, "Values": {code: label}}}
    """
    variables = parse_sps_dictionary(sps_text)
    descriptions = parse_variable_labels(sps_text)
    value_labels = parse_value_labels(sps_text)
    return {
        v.name: {
            "start": v.start,
            "end": v.end,
            "numeric": v.numeric,
            "Description": descriptions.get(v.name, ""),
            "Values": value_labels.get(v.name, {}),
        }
        for v in variables
    }


def variables_from_dictionary(
    variable_dictionary: dict[str, dict[str, object]],
) -> list[CPSVariable]:
    """Reconstruct column specs from a built/loaded variable dictionary.

    Sorted by column `start` (not dict/JSON order, which is alphabetical by
    name after save_variable_dictionary's sort_keys=True) so the resulting
    column order matches the original fixed-width record layout.
    """
    variables = [
        CPSVariable(
            name=name,
            start=int(entry["start"]),  # type: ignore[call-overload]
            end=int(entry["end"]),  # type: ignore[call-overload]
            numeric=bool(entry["numeric"]),
        )
        for name, entry in variable_dictionary.items()
    ]
    return sorted(variables, key=lambda v: v.start)


def variable_dictionary_path(
    dictionaries_dir: Path,
    year: int,
    month: int | None = None,
) -> Path:
    """Path to store dictionaries: {dictionaries_dir}/{year}{month}.json."""
    month_str = f"{month:02d}" if month is not None else ""
    return dictionaries_dir / f"{year}{month_str}.json"


def save_variable_dictionary(
    variable_dictionary: dict[str, dict[str, object]],
    dictionaries_dir: Path,
    year: int,
    month: int | None = None,
) -> Path:
    """Save to dictionary storage: data/reference/cps/{source}/{year}{month}.json."""
    out_path = variable_dictionary_path(dictionaries_dir, year, month)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(variable_dictionary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_variable_dictionary(
    dictionaries_dir: Path,
    year: int,
    month: int | None = None,
) -> dict[str, dict[str, object]]:
    """Loads a JSON-dictionary."""
    path = variable_dictionary_path(dictionaries_dir, year, month)
    result = json.loads(path.read_text(encoding="utf-8"))
    return result


def build_and_save_variable_dictionary(
    sps_path: Path,
    year: int,
    month: int | None,
    dictionaries_dir: Path,
) -> Path:
    """Build and save dictionary in one step. Added for convenience."""
    variable_dictionary = build_variable_dictionary(
        sps_path.read_text(encoding="latin-1")
    )
    return save_variable_dictionary(variable_dictionary, dictionaries_dir, year, month)


def _label_key(value: object) -> str:
    """String key to look up `value` in a {code: label} dict.

    Columns with any blank/missing cells get coerced to float64 by
    parse_fixed_width (NaN forces float).
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def apply_value_labels(
    df: pd.DataFrame, variable_dictionary: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """Replace numeric codes with their SPS value labels, for readable mart output."""
    out = df.copy()
    for column, entry in variable_dictionary.items():
        if column not in out.columns:
            continue
        labels = entry.get("Values") or {}
        if not labels:
            continue
        out[column] = out[column].map(
            lambda v, labels=labels: v if pd.isna(v) else labels.get(_label_key(v), v)
        )
    return out


def _extract_dat_text(zip_path: Path) -> str:
    """Unzip CPS data file."""
    result = subprocess.run(
        ["unzip", "-p", str(zip_path)],
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("latin-1")


def parse_fixed_width(text: str, variables: list[CPSVariable]) -> pd.DataFrame:
    """Parse fixed-width record text into a DataFrame using SPS column specs."""
    colspecs = [(v.start - 1, v.end) for v in variables]
    names = [v.name for v in variables]
    df = pd.read_fwf(io.StringIO(text), colspecs=colspecs, names=names, dtype=str)
    numeric_cols = [v.name for v in variables if v.numeric]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def parse_cps_zip(
    zip_path: Path,
    variable_dictionary: dict[str, dict[str, object]],
    year: int,
) -> pd.DataFrame:
    """Parse one CPS zip into a tidy wide DataFrame using dictionary."""
    variables = variables_from_dictionary(variable_dictionary)
    dat_text = _extract_dat_text(zip_path)
    df = parse_fixed_width(dat_text, variables)
    year_col = pd.Series(year, index=df.index, name="Year")
    df = pd.concat([year_col, df], axis=1)
    return validate_cps_long(df)


def bronze_path(bronze_dir: Path, year: int, month: int | None) -> Path:
    """Location of bronze data files."""
    month_str = "" if month is None else f"{month:02d}"
    if month is not None:
        return bronze_dir / f"{year}" / f"{year}{month_str}.parquet"
    return bronze_dir / f"{year}{month_str}.parquet"


def parse_to_bronze(
    zip_path: Path,
    variable_dictionary: dict[str, dict[str, object]],
    year: int,
    month: int | None,
    bronze_dir: Path,
) -> Path:
    """Parse and save bronze file in one step."""
    tidy = parse_cps_zip(zip_path, variable_dictionary, year)
    out_path = bronze_path(bronze_dir, year, month)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(tidy, out_path)
    return out_path
