"""Typed application settings — read from environment / .env via pydantic-settings."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env (if present) before any Settings subclass reads
# it below — makes BEA_API_KEY etc. available regardless of whether the caller's
# shell has direnv active (jobs/, pytest, Docker all just work). Never overrides
# variables already set in the real environment (load_dotenv default).
load_dotenv()


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
    def features(self) -> Path:
        return self.root / "features"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"


class Settings(BaseSettings):
    paths: DataPaths = DataPaths()
    bea_api_key: str = Field(default="", alias="BEA_API_KEY")
    ipums_api_key: str = Field(default="", alias="IPUMS_API_KEY")
    cps_mw_base_url: str = Field(
        default="https://data.nber.org/mare_winship", alias="CPS_MW_BASE_URL"
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
