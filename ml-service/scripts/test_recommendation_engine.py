"""
Celest Machine Learning Subsystem - Recommendation Validation Harness.
Instantiates the persistent recommendation service layer and validates inference.
"""

import logging
import random
import sys
import time
from typing import List, Dict, Any

from app.services.recommendation_service import RecommendationService

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("celest_recommendation_harness")

def execute_validation_test() -> None:
    """
    Instantiates the recommendation engine and validates the
    complete recommendation pipeline.
    """
    logger.info("Initializing Step 9 Recommendation Validation...")
    try:
        start_init = time.perf_counter()

        # Initialize the recommendation engine
        engine = RecommendationService()

        init_duration = time.perf_counter() - start_init

        logger.info(
            f"Recommendation engine initialized in {init_duration:.4f} seconds."
        )

    except Exception as err:
        logger.critical(
            f"Failed to initialize RecommendationService: {err}"
        )
        sys.exit(1)  
    # Get all available tracks from the manifest
    available_track_ids = list(engine.manifest_lookup.keys())

    if not available_track_ids:
        logger.error("No tracks found inside the recommendation engine.")
        sys.exit(1)

    # Pick a random song for testing
    test_track_id = random.choice(available_track_ids)
    top_k = 10

    track_info = engine.manifest_lookup[test_track_id]

    logger.info(
        f"Testing recommendations for Track ID {test_track_id} "
        f"('{track_info.get('title', 'Unknown')}' "
        f"by '{track_info.get('artist', 'Unknown')}')"
    ) 
    # Measure recommendation inference latency
    start_inference = time.perf_counter()

    try:
        recommendations: List[Dict[str, Any]] = engine.recommend(
            track_id=test_track_id,
            top_k=top_k
        )

        inference_duration = time.perf_counter() - start_inference

    except Exception as err:
        logger.error(f"Recommendation request failed: {err}")
        sys.exit(1)

    print("\n" + "=" * 120)
    print("                    CELEST RECOMMENDATION ENGINE REPORT")
    print("=" * 120)

    print(f"Initialization Latency : {init_duration * 1000:.2f} ms")
    print(f"Recommendation Latency : {inference_duration * 1000:.2f} ms")
    print(f"Query Track ID         : {test_track_id}")
    print(f"Query Title            : {track_info.get('title', 'Unknown')}")
    print(f"Query Artist           : {track_info.get('artist', 'Unknown')}")
    print(f"Top K                  : {top_k}")

    print("-" * 120)

    print(
        f"{'TRACK ID':<10} | "
        f"{'TITLE':<30} | "
        f"{'ARTIST':<25} | "
        f"{'HYBRID':<10} | "
        f"{'SEMANTIC':<10} | "
        f"{'AUDIO':<10}"
    )

    print("-" * 120)

    for item in recommendations:

        title = item.get("title", "Unknown")
        artist = item.get("artist", "Unknown")

        if len(title) > 30:
            title = title[:27] + "..."

        if len(artist) > 25:
            artist = artist[:22] + "..."

        print(
            f"{item['track_id']:<10} | "
            f"{title:<30} | "
            f"{artist:<25} | "
            f"{item['hybrid_score']:<10.6f} | "
            f"{item['semantic_score']:<10.6f} | "
            f"{item['audio_score']:<10.6f}"
        )

    print("=" * 120)

    assert len(recommendations) <= top_k, (
        "Recommendation engine returned more recommendations than requested."
    )

    logger.info("Step 9 validation completed successfully.")

if __name__ == "__main__":
    execute_validation_test()