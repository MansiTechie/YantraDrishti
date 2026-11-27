import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import ConfigurationError, settings

logger = logging.getLogger(__name__)


def create_db_engine(db_url: str):
    """
    Safely creates a SQLAlchemy engine.
    Applies SQLite-specific threading/pooling options when SQLite is used for tests.
    """
    if not db_url or not db_url.strip():
        raise ConfigurationError(
            "DATABASE_URL is not configured. Please set DATABASE_URL in your environment "
            "or .env file (e.g. postgresql+psycopg2://<user>:<password>@localhost:5432/yantradrishti "
            "or sqlite:///:memory: for isolated testing)."
        )

    connect_args = {}
    extra_kwargs = {}

    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in db_url:
            extra_kwargs["poolclass"] = StaticPool

    return create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        **extra_kwargs,
    )


# Lazily initialized primary engine and sessionmaker
_engine = None
_SessionLocal = None


def get_engine():
    """Returns the singleton engine instance, creating it if needed."""
    global _engine
    if _engine is None:
        _engine = create_db_engine(settings.DATABASE_URL)
    return _engine


def get_sessionmaker():
    """Returns the sessionmaker instance bound to the engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal


class SessionLocalProxy:
    """Proxy providing backward-compatible SessionLocal() callable semantics."""
    def __call__(self, *args, **kwargs):
        factory = get_sessionmaker()
        return factory(*args, **kwargs)


SessionLocal = SessionLocalProxy()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database session lifecycle management.
    Yields an active session and ensures it is closed after request processing.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(target_engine=None):
    """
    Development/startup table initialization helper.
    Ensures all models are registered and creates missing tables.
    Production environments should utilize Alembic migrations.
    """
    from app.db.base import Base
    import app.models.machine  # noqa: F401 - ensures MachineModel is registered

    target = target_engine or get_engine()
    Base.metadata.create_all(bind=target)
