"""
Celest Machine Learning Subsystem - Main REST API Application Server.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI

from app.routers.recommendation_router import router as recommendation_router
from app.services.recommendation_service import RecommendationService

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("celest_ml_service")


@asynccontextmanager
async def application_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Initialize the RecommendationService exactly once during application startup.
    """

    logger.info("Starting Celest ML Service...")

    try:
        app.state.recommendation_service = RecommendationService()
        logger.info("Recommendation engine initialized successfully.")

    except Exception as err:
        logger.exception("Failed to initialize RecommendationService.")
        raise RuntimeError("Application startup failed.") from err

    yield

    logger.info("Shutting down Celest ML Service...")
    app.state.recommendation_service = None


app = FastAPI(
    title="Celest ML Service",
    description="AI Engine for Audio Analysis and Hybrid Music Recommendation",
    version="1.0.0",
    lifespan=application_lifespan,
)

# Register API routers
app.include_router(recommendation_router)


@app.get("/")
async def root():
    return {
        "platform": "Celest ML Service",
        "status": "online",
        "documentation": "/docs",
        "health": "/recommendations/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Celest ML Service",
    }