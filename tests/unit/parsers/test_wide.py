import pandas as pd

from src.parsers.bea.wide import to_wide


def _long_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LineNumber": [1, 1, 2, 2, 3, 3],
            "LineDescription": ["A", "A", "B", "B", "C", "C"],
            "Year": [2020, 2021, 2020, 2021, 2020, 2021],
            "DataValue": [10.0, 11.0, 20.0, 22.0, 30.0, 33.0],
        }
    )


def test_pivots_to_line_by_year() -> None:
    dim_table = pd.DataFrame({"LineNumber": [1, 2, 3]})
    wide = to_wide(_long_df(), dim_table)

    assert list(wide.index) == [1, 2, 3]
    assert list(wide.columns) == [2020, 2021]
    assert wide.loc[1, 2020] == 10.0
    assert wide.loc[2, 2021] == 22.0


def test_excludes_lines_absent_from_dim_table() -> None:
    dim_table = pd.DataFrame({"LineNumber": [1, 2]})  # line 3 not included
    wide = to_wide(_long_df(), dim_table)

    assert 3 not in wide.index
    assert set(wide.index) == {1, 2}


def test_axis_names_cleared() -> None:
    dim_table = pd.DataFrame({"LineNumber": [1]})
    wide = to_wide(_long_df(), dim_table)

    assert wide.index.name == "LineNumber"
    assert wide.columns.name is None
