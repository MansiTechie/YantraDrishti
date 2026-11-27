import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from repository root or backend directory if present
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


class ConfigurationError(ValueError):
    """Raised when critical configuration is missing or invalid."""
    pass


class Settings:
    PROJECT_NAME: str = "YantraDrishti API"
    DESCRIPTION: str = "AI-Powered Predictive Digital Twin for Smart Manufacturing"
    VERSION: str = "0.1.0"

    # Optional explicit model artifact path override
    MODEL_ARTIFACT_PATH: str | None = os.getenv("MODEL_ARTIFACT_PATH", None)

    @property
    def DATABASE_URL(self) -> str:
        """
        Retrieves DATABASE_URL from environment.
        Raises ConfigurationError if DATABASE_URL is unset or empty,
        preventing silent fallback to hardcoded or insecure credentials.
        """
        url = os.getenv("DATABASE_URL", "").strip()
        if not url:
            raise ConfigurationError(
                "DATABASE_URL is not configured. Please set the DATABASE_URL environment "
                "variable or define it in your .env file.\n"
                "Example for PostgreSQL: DATABASE_URL=postgresql+psycopg2://<user>:<password>@localhost:5432/yantradrishti\n"
                "Example for isolated testing: DATABASE_URL=sqlite:///:memory:"
            )
        return url


settings = Settings()
