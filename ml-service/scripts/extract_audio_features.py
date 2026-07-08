"""
Celest Machine Learning Subsystem - Feature Extraction Orchestrator.
Iterates over catalog_manifest.json entries and processes audio files one by one.
"""

import json
import logging
import gc
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config.dataset_config import dataset_config
from app.services.audio_feature_service import AudioFeatureService

# Configure structured, production-ready logging outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("celest_audio_feature_pipeline")

FEATURE_VERSION = "1.0.0"


def execute_extraction_pipeline() -> None:
    """Runs the feature extraction pipeline on unregistered tracks in the manifest."""
    logger.info("Starting Celest Audio Feature Extraction Engine...")

    manifest_path = dataset_config.dataset_metadata_dir / "catalog_manifest.json"
    output_base_dir = dataset_config.dataset_processed_audio_dir

    if not manifest_path.exists():
        logger.error(f"Execution Error: Base manifest file missing at: {manifest_path}")
        return

    # Ingest baseline structural data manifest securely
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_state = json.load(f)

    tracks = manifest_state.get("tracks", [])
    total_tracks = len(tracks)
    logger.info(f"Loaded master manifest index. Found {total_tracks} total track entries.")

    feature_service = AudioFeatureService()
    run_summary = {"processed": 0, "skipped": 0, "failed": 0}
    
    start_time = time.perf_counter()

    for track in tracks:
        # Check if features have already been processed to maintain pipeline idempotency
        if track.get("features_generated") is True:
            run_summary["skipped"] += 1
            continue

        track_id = track.get("track_id")
        unique_id = track.get("id")
        relative_path_str = track.get("relative_path", "")

        # Compute dynamic location independent of previous machine-specific absolute roots
        audio_path = dataset_config.dataset_raw_audio_dir / relative_path_str

        if not audio_path.exists():
            logger.warning(f"[-] File missing on disk for Track ID {track_id} at: {audio_path}")
            run_summary["failed"] += 1
            continue

        try:
            # Extract acoustic feature arrays
            extracted_features = feature_service.extract_track_features(audio_path)

            # Determine target file destination matching raw nested paths (e.g. 000/000002.json)
            relative_path_obj = Path(relative_path_str)
            target_subfolder = output_base_dir / relative_path_obj.parent
            target_subfolder.mkdir(parents=True, exist_ok=True)
            
            target_json_path = target_subfolder / f"{relative_path_obj.stem}.json"

            # Build single discrete tracking node payload
            feature_payload = {
                "id": unique_id,
                "track_id": track_id,
                "feature_version": FEATURE_VERSION,
                "feature_extraction_timestamp": datetime.now(UTC).isoformat(),
                **extracted_features
            }

            # Serialize feature JSON to disk, preserving Unicode strings
            with open(target_json_path, "w", encoding="utf-8") as out_file:
                json.dump(feature_payload, out_file, indent=4, ensure_ascii=False)

            # Update track node status in memory
            track["features_generated"] = True
            run_summary["processed"] += 1

            # Proactive memory cleanup inside long loops
            del extracted_features
            del feature_payload
            if run_summary["processed"] % 500 == 0:
                gc.collect()

            # Periodic logging loop (fires at every 100-track boundary)
            if run_summary["processed"] % 100 == 0:
                logger.info(f"[*] Extraction progress checkpoint: Extracted features for {run_summary['processed']} tracks.")

        except Exception as err:
            logger.error(f"[!] Critical extraction fault on Track ID {track_id} ({audio_path.name}): {err}")
            run_summary["failed"] += 1
            continue

    end_time = time.perf_counter()
    total_runtime = end_time - start_time

    # Commit manifest status modifications to disk if processing occurred
    if run_summary["processed"] > 0:
        logger.info("Committing pipeline execution updates back to catalog_manifest.json...")
        try:
            with open(manifest_path, "w", encoding="utf-8") as out_manifest:
                json.dump(manifest_state, out_manifest, indent=4, ensure_ascii=False)
        except Exception as fatal_err:
            logger.error(f"Fatal write exception mutating main ledger asset: {fatal_err}")
            return

    # Calculate runtime metrics
    avg_processing_time = (total_runtime / run_summary["processed"]) if run_summary["processed"] > 0 else 0.0

    # Print final execution metrics report
    print("\n" + "=" * 50)
    print("       AUDIO FEATURE EXTRACTION RUN SUMMARY")
    print("=" * 50)
    print(f" Tracks Processed:     {run_summary['processed']}")
    print(f" Tracks Skipped:       {run_summary['skipped']}")
    print(f" Processing Failures:  {run_summary['failed']}")
    print("-" * 50)
    print(f" Total Runtime:        {total_runtime:.2f} seconds")
    print(f" Avg Time Per Track:   {avg_processing_time:.4f} seconds")
    print("=" * 50)
    logger.info("🎉 Step 6 execution completed successfully.")


if __name__ == "__main__":
    execute_extraction_pipeline()