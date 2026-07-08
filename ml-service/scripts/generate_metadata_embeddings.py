"""
Celest Machine Learning Subsystem - Metadata Embedding Pipeline Orchestrator.
Iterates over catalog_manifest.json entries and processes text strings one by one.
"""

import json
import logging
import gc
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config.dataset_config import dataset_config
from app.services.metadata_embedding_service import MetadataEmbeddingService

# Centralized Pipeline Configuration Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_VERSION = "1.0.0"

# Configure structured, production-ready logging outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("celest_metadata_embedding_pipeline")


def execute_embedding_pipeline() -> None:
    """Runs the text embedding generation pipeline based on manifest tracking parameters."""
    logger.info("Starting Celest Metadata Embedding Extraction Engine...")

    manifest_path = dataset_config.dataset_metadata_dir / "catalog_manifest.json"
    
    # Target output path context programmatically pointing to the renamed directory
    output_base_dir = dataset_config.dataset_processed_metadata_dir

    if not manifest_path.exists():
        logger.error(f"Execution Error: Base manifest file missing at: {manifest_path}")
        return

    # Ingest baseline structural data manifest securely
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_state = json.load(f)

    tracks = manifest_state.get("tracks", [])
    total_tracks = len(tracks)
    logger.info(f"Loaded master manifest index. Found {total_tracks} total track entries.")

    # Initialize the core service layer exactly once at startup
    embedding_service = MetadataEmbeddingService(
        metadata_dir=dataset_config.dataset_metadata_dir,
        model_name=EMBEDDING_MODEL
    )
    
    run_summary = {"processed": 0, "skipped": 0, "failed": 0}
    checkpoint_pending_saves = False
    start_time = time.perf_counter()

    for track in tracks:
        track_id = track.get("track_id")
        unique_id = track.get("id")
        relative_path_str = track.get("relative_path", "")

        # Compute expected output path before analyzing state to safeguard against checkpoint drops
        relative_path_obj = Path(relative_path_str)
        target_subfolder = output_base_dir / relative_path_obj.parent
        target_json_path = target_subfolder / f"{relative_path_obj.stem}.json"

        # Double-Layer Idempotency Guard: Manifest Check OR Direct File Presence Lookups
        if track.get("embeddings_generated") is True or target_json_path.exists():
            # If file is on disk but manifest didn't checkpoint yet, catch up the in-memory state
            if not track.get("embeddings_generated"):
                track["embeddings_generated"] = True
                checkpoint_pending_saves = True
                
            run_summary["skipped"] += 1
            continue

        try:
            # 1. Map relational data tokens into an enriched text document
            text_document = embedding_service.build_metadata_document(track)
            
            # 2. Convert text document into a 384-dimensional dense float vector
            vector_embedding = embedding_service.generate_embedding(text_document)

            # Ensure destination subfolders match the original raw audio splits
            target_subfolder.mkdir(parents=True, exist_ok=True)

            # Build single discrete tracking node payload containing text document context
            embedding_payload = {
                "id": unique_id,
                "track_id": track_id,
                "embedding_version": EMBEDDING_VERSION,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "metadata_document": text_document,
                "embedding_generation_timestamp": datetime.now(UTC).isoformat(),
                "embedding": vector_embedding
            }

            # Serialize embedding JSON to disk, preserving Unicode strings
            with open(target_json_path, "w", encoding="utf-8") as out_file:
                json.dump(embedding_payload, out_file, indent=4, ensure_ascii=False)

            # Update track node status in memory
            track["embeddings_generated"] = True
            run_summary["processed"] += 1
            checkpoint_pending_saves = True

            # Proactive memory cleanup inside long execution loops
            del text_document
            del vector_embedding
            del embedding_payload
            
            if run_summary["processed"] % 500 == 0:
                gc.collect()

            # Periodic checkpoint interval loop (fires at every 100 processed tracks)
            if run_summary["processed"] % 100 == 0:
                logger.info(f"[*] Embedding progress checkpoint: Calculated embeddings for {run_summary['processed']} tracks. Persisting manifest...")
                with open(manifest_path, "w", encoding="utf-8") as out_manifest:
                    json.dump(manifest_state, out_manifest, indent=4, ensure_ascii=False)
                checkpoint_pending_saves = False

        except Exception as err:
            logger.error(f"[!] Critical embedding processing fault on Track ID {track_id}: {err}")
            run_summary["failed"] += 1
            continue

    end_time = time.perf_counter()
    total_runtime = end_time - start_time

    # Commit final manifest state modifications to disk if updates remain un-checkpointed
    if checkpoint_pending_saves or run_summary["processed"] > 0:
        logger.info("Committing final pipeline execution updates back to catalog_manifest.json...")
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
    print("       METADATA EMBEDDING PIPELINE RUN SUMMARY")
    print("=" * 50)
    print(f" Tracks Processed:     {run_summary['processed']}")
    print(f" Tracks Skipped:       {run_summary['skipped']}")
    print(f" Processing Failures:  {run_summary['failed']}")
    print("-" * 50)
    print(f" Total Runtime:        {total_runtime:.2f} seconds")
    print(f" Avg Time Per Track:   {avg_processing_time:.4f} seconds")
    print("=" * 50)
    logger.info("🎉 Step 7 execution completed successfully.")


if __name__ == "__main__":
    execute_embedding_pipeline()