"""Typed application settings — read from environment / .env via pydantic-settings."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env (if present) before any Settings subclass reads
# it below:
#  - Makes BEA_API_KEY etc. available regardless of whether the caller's
#    shell has direnv active.
#  - Never overrides variables already set in the real environment.
load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]


class DataPaths(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_")

    root: Path = Field(default=Path.home() / "projects/econ-ge/data")

    @property
    def external(self) -> Path:
        return self.root / "external"

    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def reference(self) -> Path:
        return self.root / "reference"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def cps_external_dir(self, source: str) -> Path:
        """Raw zip/SPS storage for a CPS source, e.g. data/external/cps/basic/."""
        return self.external / "cps" / source

    def cps_raw_dictionaries_dir(self, source: str) -> Path:
        """Raw .sps dictionaries for a CPS source, e.g. data/external/cps/mw/dictionaries/."""
        return self.cps_external_dir(source) / "dictionaries"

    def cps_bronze_dir(self, source: str) -> Path:
        """Bronze parquet root for a CPS source, e.g. data/bronze/cps/basic/."""
        return self.bronze / "cps" / source

    def cps_clean_dictionaries_dir(self, source: str) -> Path:
        """Cleaned JSON variable dictionaries for a CPS source, e.g. data/reference/cps/mw/."""
        return self.reference / "cps" / source

    def ipums_bronze_dir(self, collection: str) -> Path:
        """Bronze parquet root for an IPUMS collection, e.g. data/bronze/ipums/cps/."""
        return self.bronze / "ipums" / collection

    def ipums_clean_dictionaries_dir(self, collection: str) -> Path:
        """Cleaned JSON variable dictionaries for an IPUMS collection, e.g. data/reference/ipums/cps/."""
        return self.reference / "ipums" / collection


class Settings(BaseSettings):
    paths: DataPaths = DataPaths()
    # Sibling of data/, not nested under DataPaths - holds versioned YAML +
    # crosswalk config for the src/cleaning/ methodology layer
    # (CleaningContext.from_config), not raw/bronze/silver data.
    cleaning_config_root: Path = Field(
        default=_REPO_ROOT / "config" / "cleaning",
        alias="CLEANING_CONFIG_ROOT",
    )
    # Sibling of cleaning_config_root - the expected bronze column set per
    # collection, read by src/config/parsing.py.
    parsing_config_root: Path = Field(
        default=_REPO_ROOT / "config" / "parsing",
        alias="PARSING_CONFIG_ROOT",
    )
    bea_api_key: str = Field(default="", alias="BEA_API_KEY")
    ipums_api_key: str = Field(default="", alias="IPUMS_API_KEY")
    cps_mw_base_url: str = Field(
        default="https://data.nber.org/mare_winship", alias="CPS_MW_BASE_URL"
    )
    cps_basic_base_url: str = Field(
        default="https://data.nber.org/cps-basic3/dat/", alias="CPS_BASIC_BASE_URL"
    )
    cps_basic_sps_base_url: str = Field(
        default="https://data.nber.org/cps-basic3/programs/",
        alias="CPS_BASIC_SPS_BASE_URL",
    )
    postgres_user: str = Field(default="", alias="POSTGRES_USER")
    postgres_password: str = Field(default="", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="", alias="POSTGRES_DB")
    postgres_host_port: int = Field(default=5432, alias="POSTGRES_HOST_PORT")

    @property
    def postgres_connection_params(self) -> dict:
        return {
            "host": "localhost",
            "port": self.postgres_host_port,
            "dbname": self.postgres_db,
            "user": self.postgres_user,
            "password": self.postgres_password,
        }


settings = Settings()
