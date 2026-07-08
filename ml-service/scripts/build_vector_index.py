"""
Celest Machine Learning Subsystem - Vector Index Compilation Orchestrator.
Loads structured text embeddings sequentially and assembles exact nearest-neighbour indices.
"""

import logging
import sys
import time
import numpy as np

from app.config.dataset_config import dataset_config
from app.services.vector_index_service import VectorIndexService

# Configure structured, production-ready logging outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("celest_vector_index_pipeline")


def run_index_construction(force_rebuild: bool = False) -> None:
    """
    Orchestrates the vector aggregation process. Extracts properties from individual files
    and populates structural FAISS binary maps.
    """
    logger.info("Starting Celest Vector Index Builder System...")

    output_dir = dataset_config.dataset_vector_index_dir
    faiss_file = output_dir / "metadata_index.faiss"
    mapping_file = output_dir / "metadata_track_mapping.json"
    metadata_file = output_dir / "metadata_index_metadata.json"

    # Enforce strict build checks to avoid accidental file mutations
    artifacts_exist = faiss_file.exists() and mapping_file.exists() and metadata_file.exists()
    if artifacts_exist and not force_rebuild:
        logger.info("[-] Vector Index components already exist on disk. Pipeline execution skipped.")
        logger.info("💡 To recreate or overwrite these files, pass 'force_rebuild=True' into the pipeline runtime.")
        return

    if artifacts_exist and force_rebuild:
        logger.warning("[!] 'force_rebuild=True' passed. Cleaning up previous index files...")
        for file_target in [faiss_file, mapping_file, metadata_file]:
            if file_target.exists():
                file_target.unlink()
                logger.info(f" -> Removed old artifact: {file_target.name}")

    embedding_source = dataset_config.dataset_processed_metadata_dir
    index_service = VectorIndexService()

    # Discover file targets recursively
    embedding_files = index_service.scan_embedding_files(embedding_source)
    total_discovered = len(embedding_files)
    logger.info(f"Discovered {total_discovered} processing targets inside storage cache.")

    if total_discovered == 0:
        logger.error("❌ Aborting Construction: No embedding arrays discovered. Run step 7 first.")
        return

    valid_track_ids = []
    vector_accumulation_list = []
    metrics = {"processed": 0, "skipped": 0, "failures": 0}

    start_perf_time = time.perf_counter()

    # Iterate over individual embedding files sequentially
    for file_path in embedding_files:
        track_id, vector = index_service.validate_and_extract(file_path)
        
        if track_id is None or vector is None:
            metrics["skipped"] += 1
            metrics["failures"] += 1
            continue

        valid_track_ids.append(track_id)
        vector_accumulation_list.append(vector)
        metrics["processed"] += 1

        # Periodic logging checkpoint loop (fires at every 500 tracks processed)
        if metrics["processed"] % 500 == 0:
            logger.info(f"[*] Parsing progress checkpoint: Validated and parsed {metrics['processed']} vectors.")

    logger.info(f"Finished validation checks. Valid vectors: {metrics['processed']} | Skipped vectors: {metrics['skipped']}")

    if metrics["processed"] == 0:
        logger.error("❌ Operational Error: Zero valid float32 vectors gathered. Index generation terminated.")
        return

    logger.info("Compiling floating-point NumPy matrix block layout...")
    # Convert accumulation list directly to a highly continuous contiguous float32 block layout
    np_matrix = np.asarray(vector_accumulation_list, dtype=np.float32)

    logger.info("Building FAISS Inner Product Index matrix structure...")
    index = index_service.build_faiss_index(np_matrix)

    logger.info(
        f"FAISS Index Statistics | "
        f"Vectors: {index.ntotal} | "
        f"Dimension: {index.d}"
    )

    logger.info("Writing index artifacts down to disk storage space...")
    index_service.serialize_artifacts(
        output_dir=output_dir,
        index=index,
        track_mapping=valid_track_ids,
        vector_count=metrics["processed"]
    )

    end_perf_time = time.perf_counter()
    total_runtime = end_perf_time - start_perf_time
    avg_processing_time = (total_runtime / metrics["processed"]) if metrics["processed"] > 0 else 0.0

    # Print final operational execution report metrics
    print("\n" + "=" * 55)
    print("           VECTOR INDEX BUILD SUMMARY")
    print("=" * 55)
    print(f" Vectors Indexed:       {metrics['processed']}")
    print(f" Vectors Skipped:       {metrics['skipped']}")
    print(f" Parsing Failures:      {metrics['failures']}")
    print("-" * 55)
    print(f" Total Pipeline Runtime: {total_runtime:.4f} seconds")
    print(f" Avg Time Per Vector:   {avg_processing_time:.6f} seconds")
    print("=" * 55)
    logger.info("🎉 Step 8 Vector Index Construction completed successfully.")


if __name__ == "__main__":
    # Standard runtime fallback defaults to force_rebuild=False
    # To trigger a full database wipe and rebuild, flip parameter explicitly to True here
    run_index_construction(force_rebuild=False)