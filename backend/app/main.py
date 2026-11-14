
from fastapi import FastAPI

from app.api.machines import router as machines_router

app = FastAPI(
    title="YantraDrishti API",
    description="AI-Powered Predictive Digital Twin for Smart Manufacturing",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "project": "YantraDrishti",
        "message": "Backend is running",
        "version": "0.1.0",
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "YantraDrishti Backend",
    }


app.include_router(machines_router)
