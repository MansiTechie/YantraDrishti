from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from sqlalchemy.exc import DBAPIError, OperationalError

from app.api.machines import router as machines_router
from app.api.predictions import router as predictions_router
from app.core.config import settings
from app.db.session import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on application startup
    try:
        init_db()
        logger.info("Database tables verified/initialized successfully.")
    except (OperationalError, DBAPIError) as exc:
        logger.warning(
            "Database table initialization skipped during startup: %s. "
            "If using PostgreSQL, ensure the server is running and accessible.",
            exc,
        )
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "project": "YantraDrishti",
        "message": "Backend is running",
        "version": settings.VERSION,
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "YantraDrishti Backend",
    }


app.include_router(machines_router)
app.include_router(predictions_router)
