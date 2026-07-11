"""Bronze-layer parsing of raw CPS Mare-Winship extractor files into a tidy
DataFrame, persisted as bronze parquet.

Reads the as-saved zip (fixed-width data file) from
extractors.cps_mw.CPSMWExtractor plus its covering SPS dictionary, and
produces one row per person-record with one typed column per SPS variable.
Kept separate from extraction so it's testable on small in-memory fixed-width
strings without hitting the network.

SPS dictionary format (NBER's Mare-Winship files use SPSS `input program` /
`data list file=... /` syntax, confirmed against the actual
cpsmw64_88.sps/cpsmw89_92.sps dictionaries): one variable per line,
`name start-end` (1-indexed, inclusive column range; a single trailing number
means a 1-column field) with variables ending at a lone "." line, e.g.:

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

A trailing `(a)` marks a variable alphanumeric (string); everything else is
numeric (SPS's default when no format is given — unlike BEA's SPS dialect,
there's no explicit `(F...)` marker here). The `variable labels` block
(variable name -> human-readable description, one line per variable, no `/`
separators needed since there's only one description per line) and the
`value labels` block (variable name -> {numeric code: value label}, one
variable's codes per `/`-separated group) are both parsed — by
parse_variable_labels() and parse_value_labels() below — and combined by
build_variable_dictionary() into the per-year JSON dictionaries under
parsers/dictionaries/cpsmw/, each variable stored as
{"Description": ..., "Values": {code: label}}.
"""

import io
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.input_output.parquet import write_parquet
from src.schemas.bronze.cps_mw_long import validate_cps_mw_long

DICTIONARIES_DIR = Path(__file__).resolve().parent / "dictionaries" / "cpsmw"

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
class CPSMWVariable:
    name: str
    start: int  # 1-indexed, inclusive
    end: int  # 1-indexed, inclusive
    numeric: bool


def parse_sps_dictionary(sps_text: str) -> list[CPSMWVariable]:
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
            CPSMWVariable(
                name=m.group("name"),
                start=start,
                end=end,
                numeric=m.group("fmt") is None,
            )
        )
    if not variables:
        raise ValueError("No variable definitions found in SPS dictionary")
    return variables


def load_sps_dictionary(sps_path: Path) -> list[CPSMWVariable]:
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
    """Combine parse_variable_labels + parse_value_labels into the JSON shape
    persisted by save_variable_dictionary:

        {variable_name: {"Description": str, "Values": {code: label}}}

    Every variable that has a description and/or value labels gets an entry;
    a variable with no value labels defined (common for continuous
    variables, e.g. a weight or an ID) still gets a "Values" key, just an
    empty dict, so callers can always index it without a KeyError.
    """
    descriptions = parse_variable_labels(sps_text)
    value_labels = parse_value_labels(sps_text)
    variable_names = set(descriptions) | set(value_labels)
    return {
        name: {
            "Description": descriptions.get(name, ""),
            "Values": value_labels.get(name, {}),
        }
        for name in sorted(variable_names)
    }


def variable_dictionary_path(
    year: int, dictionaries_dir: Path = DICTIONARIES_DIR
) -> Path:
    """Where a year's variable dictionary lives: {dictionaries_dir}/cpsmw_{year}.json."""
    return dictionaries_dir / f"cpsmw_{year}.json"


def save_variable_dictionary(
    variable_dictionary: dict[str, dict[str, object]],
    year: int,
    dictionaries_dir: Path = DICTIONARIES_DIR,
) -> Path:
    """Persist a built variable dictionary as parsers/dictionaries/cpsmw/cpsmw_{year}.json."""
    out_path = variable_dictionary_path(year, dictionaries_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(variable_dictionary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_variable_dictionary(
    year: int, dictionaries_dir: Path = DICTIONARIES_DIR
) -> dict[str, dict[str, object]]:
    """Read back a year's variable dictionary saved by save_variable_dictionary."""
    path = variable_dictionary_path(year, dictionaries_dir)
    result: dict[str, dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    return result


def build_and_save_variable_dictionary(
    sps_path: Path, year: int, dictionaries_dir: Path = DICTIONARIES_DIR
) -> Path:
    """build_variable_dictionary() an SPS file on disk + save_variable_dictionary() in one step."""
    variable_dictionary = build_variable_dictionary(
        sps_path.read_text(encoding="latin-1")
    )
    return save_variable_dictionary(variable_dictionary, year, dictionaries_dir)


def _label_key(value: object) -> str:
    """String key to look up `value` in a {code: label} dict.

    Columns with any blank/missing cells get coerced to float64 by
    parse_fixed_width (NaN forces float), so an originally-integer code like
    5 can arrive here as 5.0 — normalize whole-number floats back to "5" so
    they still match the SPS dictionary's plain-digit string keys.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def apply_value_labels(
    df: pd.DataFrame, variable_dictionary: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """Replace numeric codes with their SPS value labels, for readable mart output.

    `variable_dictionary` is the {"Description": ..., "Values": {code:
    label}} shape built by build_variable_dictionary / loaded via
    load_variable_dictionary. Returns a new DataFrame; `df` is untouched.
    Only columns present in both `df` and `variable_dictionary` are touched,
    and only if that variable's "Values" is non-empty. A code with no entry
    in the dictionary (common for continuous variables like income, which
    are never fully enumerated) or a missing value is left as-is rather than
    becoming NaN — this is a display transform, not a re-validation.
    """
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
    """Pull the single data member out of a Mare-Winship zip as text.

    Shells out to the system `unzip -p` (streams the sole member to stdout)
    rather than stdlib zipfile: NBER's archives were built with a zip tool old
    enough that Python's zipfile rejects their "extra field" data as corrupt,
    while `unzip`/`7z` read them fine. The archived member also isn't named
    `*.dat` (it's a plain filename like `cpsmw64`), so `-p` sidesteps needing
    to know the member name at all.
    """
    result = subprocess.run(
        ["unzip", "-p", str(zip_path)],
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("latin-1")


def parse_fixed_width(text: str, variables: list[CPSMWVariable]) -> pd.DataFrame:
    """Parse fixed-width record text into a DataFrame using SPS column specs."""
    colspecs = [(v.start - 1, v.end) for v in variables]
    names = [v.name for v in variables]
    df = pd.read_fwf(io.StringIO(text), colspecs=colspecs, names=names, dtype=str)
    numeric_cols = [v.name for v in variables if v.numeric]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def parse_cps_mw_zip(zip_path: Path, sps_path: Path, year: int) -> pd.DataFrame:
    """Parse one CPS Mare-Winship zip + SPS dictionary into a tidy wide DataFrame.

    Columns out: Year (int) followed by one column per SPS variable.
    """
    variables = load_sps_dictionary(sps_path)
    dat_text = _extract_dat_text(zip_path)
    df = parse_fixed_width(dat_text, variables)
    year_col = pd.Series(year, index=df.index, name="Year")
    df = pd.concat([year_col, df], axis=1)
    return validate_cps_mw_long(df)


def bronze_path(bronze_dir: Path, year: int) -> Path:
    """Where a CPS-MW year's bronze parquet lives: {bronze_dir}/mw/{year}.parquet."""
    return bronze_dir / "mw" / f"{year}.parquet"


def parse_to_bronze(
    zip_path: Path,
    sps_path: Path,
    year: int,
    bronze_dir: Path,
) -> Path:
    """Parse one external CPS-MW zip and persist it as its own bronze parquet.

    One year's zip in -> one parquet file out: a parsing bug or re-run for one
    year doesn't touch the other years' bronze files.
    """
    tidy = parse_cps_mw_zip(zip_path, sps_path, year)
    out_path = bronze_path(bronze_dir, year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(tidy, out_path)
    return out_path
