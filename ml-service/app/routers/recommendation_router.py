"""
Celest Machine Learning Subsystem - Recommendation API Routers.
Exposes high-performance, type-hinted REST endpoints bound to the single service memory context.
"""

import logging
import time
from fastapi import APIRouter, HTTPException, Query, Request, status

from app.models.recommendation_models import (
    RecommendationResponse,
    HealthResponse,
    StatsResponse,
    QueryTrack,
    RecommendationItem
)

logger = logging.getLogger("celest_api_router")

router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations Engine"]
)


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def check_engine_health(request: Request) -> HealthResponse:
    """
    Evaluates engine memory state variables to report direct pipeline diagnostic parameters.
    Returns HTTP 500 if the single runtime service module context layer fails lookup tracking.
    """
    try:
        engine = request.app.state.recommendation_service
        if engine is None or engine.index is None:
            raise AttributeError("Inference engine instance properties are corrupted or offline.")
            
        return HealthResponse(
            status="healthy",
            engine_loaded=True,
            vectors=int(engine.index.ntotal),
            tracks=len(engine.manifest_lookup),
            cached_audio_features=len(engine.audio_feature_cache)
        )
    except Exception as err:
        logger.error(f"[!] System health verification exception trace: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The hybrid recommendation inference service singleton state is currently unhydrated or corrupted."
        )


@router.get("/stats", response_model=StatsResponse, status_code=status.HTTP_200_OK)
async def retrieve_engine_statistics(request: Request) -> StatsResponse:
    """
    Exposes granular metrics describing memory footprints, dimensions, and static configuration thresholds.
    """
    try:
        engine = request.app.state.recommendation_service
        return StatsResponse(
            vectors=int(engine.index.ntotal),
            track_mapping=len(engine.track_mapping),
            audio_cache=len(engine.audio_feature_cache),
            embedding_dimension=int(engine.index.d),
            candidate_pool_size=int(engine.__class__.CANDIDATE_POOL_SIZE)
        )
    except Exception as err:
        logger.error(f"[!] Analytics data compilation failure: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error resolving machine learning framework metrics cache profiles."
        )


@router.get("/{track_id}", response_model=RecommendationResponse, status_code=status.HTTP_200_OK)
async def fetch_hybrid_recommendations(
    track_id: int,
    request: Request,
    top_k: int = Query(default=10, ge=1, le=100, description="The number of hybrid ranked elements to return (1-100).")
) -> RecommendationResponse:
    """
    Computes real-time top-K multi-modal predictions for an indexed track ID anchor.
    Returns HTTP 404 if the requested identity matches an unknown or omitted track signature.
    """
    start_perf_time = time.perf_counter()
    logger.info(f"Incoming recommendation query -> Path ID: {track_id} | Parameters -> Top-K: {top_k}")
    engine = request.app.state.recommendation_service
    
    # 1. Structural Identity Mapping Validation Pass
    if track_id not in engine.manifest_lookup:
        latency_ms = (time.perf_counter() - start_perf_time) * 1000
        logger.warning(f"[-] Request Rejected [Latency: {latency_ms:.2f} ms]: Target track query reference [{track_id}] is unknown inside manifest maps.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track lookup failed. Requested reference mapping ID '{track_id}' is not indexed inside Celest catalog logs."
        )

    try:
        # 2. Query Single Shared Memory Service Context
        raw_recommendations = engine.recommend(track_id=track_id, top_k=top_k)
        latency_ms = (time.perf_counter() - start_perf_time) * 1000
        logger.info(f"[+] Request processed successfully. Engine Inference Latency: {latency_ms:.2f} ms")

        # 3. Construct Unified Response DTO Components
        source_meta = engine.manifest_lookup[track_id]
        query_track_dto = QueryTrack(
            track_id=track_id,
            title=source_meta.get("title") or "Unknown",
            artist=source_meta.get("artist") or "Unknown",
            album=source_meta.get("album") or "Unknown"
        )

        recommendation_items = [
            RecommendationItem(
                track_id=int(item["track_id"]),
                title=str(item["title"]),
                artist=str(item["artist"]),
                album=str(item["album"]),
                hybrid_score=float(item["hybrid_score"]),
                semantic_score=float(item["semantic_score"]),
                audio_score=float(item["audio_score"])
            )
            for item in raw_recommendations
        ]

        return RecommendationResponse(
            query_track=query_track_dto,
            recommendations=recommendation_items
        )

    except Exception as err:
        logger.error(f"[!] Real-time query inference pipeline fault: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected exception encountered inside the hybrid modeling execution loop."
        )