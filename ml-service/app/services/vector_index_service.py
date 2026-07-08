"""
Celest Machine Learning Subsystem - Vector Index Service.
Handles structural loading, NumPy translation, matrix compilation, and FAISS indexing.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import faiss
import numpy as np

logger = logging.getLogger("celest_vector_index_service")


class VectorIndexService:
    """
    Service layer responsible for parsing single-track embedding JSON structures,
    validating array schemas, and compiling high-performance FAISS indices.
    """

    def __init__(self, embedding_dimension: int = 384, expected_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes core constraints for index verification filters.
        """
        self.embedding_dimension = embedding_dimension
        self.expected_model = expected_model

    def scan_embedding_files(self, base_embedding_dir: Path) -> List[Path]:
        """
        Recursively discovers all embedding JSON targets inside the processed folder structure.
        Bypasses raw directory walking loops by gathering matching suffix signatures.
        """
        if not base_embedding_dir.exists():
            logger.error(f"Target embedding source directory missing: {base_embedding_dir}")
            return []

        return sorted(base_embedding_dir.rglob("*.json"))

    def validate_and_extract(self, file_path: Path) -> Tuple[Optional[int], Optional[List[float]]]:
        """
        Loads a single embedding JSON file and enforces shape and model constraints.
        Returns a tuple of (track_id, embedding_vector) if valid, otherwise (None, None).
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            track_id = data.get("track_id")
            vector = data.get("embedding")
            model = data.get("embedding_model")
            dimension = data.get("embedding_dimension")

            if track_id is None or vector is None:
                logger.warning(f"[-] Validation Skip: Missing core properties inside payload: {file_path.name}")
                return None, None

            if model != self.expected_model:
                logger.warning(f"[-] Model Mismatch: Expected '{self.expected_model}', found '{model}' inside {file_path.name}")
                return None, None

            if dimension != self.embedding_dimension or len(vector) != self.embedding_dimension:
                logger.warning(f"[-] Dimensionality Anomaly: Expected dimension {self.embedding_dimension}, found {len(vector)} inside {file_path.name}")
                return None, None

            vector_np = np.asarray(vector, dtype=np.float32)

            if not np.isfinite(vector_np).all():
                logger.warning(f"[-] Invalid numeric values found in {file_path.name}")
                return None, None

            return int(track_id), vector

        except (json.JSONDecodeError, TypeError, ValueError) as err:
            logger.error(f"[!] Malformed file payload or corruption encountered inside {file_path.name}: {err}")
            return None, None

    def build_faiss_index(self, embedding_matrix: np.ndarray) -> faiss.IndexFlatIP:
        """
        Instantiates a flat inner product index and registers the array matrix.
        Matrix entries must be explicitly pre-cast to contiguous C-ordered float32 types.
        """
        if embedding_matrix.dtype != np.float32:
            embedding_matrix = embedding_matrix.astype(np.float32)

        if not embedding_matrix.flags['C_CONTIGUOUS']:
            embedding_matrix = np.ascontiguousarray(embedding_matrix)

        dimension = embedding_matrix.shape[1]
        if dimension != self.embedding_dimension:
            raise ValueError(f"Matrix dimension {dimension} violates index rules ({self.embedding_dimension})")

        # Create Flat Inner Product index architecture
        index = faiss.IndexFlatIP(dimension)
        index.add(embedding_matrix)
        return index

    def serialize_artifacts(
        self, 
        output_dir: Path, 
        index: faiss.IndexFlatIP, 
        track_mapping: List[int], 
        vector_count: int
    ) -> None:
        """
        Writes the FAISS index, lookups, and operational metadata profiles securely to disk.
        Enforces ensure_ascii=False constraints for JSON serializations.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss_out_path = output_dir / "metadata_index.faiss"
        mapping_out_path = output_dir / "metadata_track_mapping.json"
        metadata_out_path = output_dir / "metadata_index_metadata.json"

        # 1. Serialize physical FAISS binary stream array
        faiss.write_index(index, str(faiss_out_path.resolve()))
        logger.info(f"[+] Serialized FAISS Vector Index binary file: {faiss_out_path.name}")

        # 2. Serialize contiguous zero-indexed tracking row lookups
        with open(mapping_out_path, "w", encoding="utf-8") as map_f:
            json.dump(track_mapping, map_f, indent=4, ensure_ascii=False)
        logger.info(f"[+] Serialized Track Translation Mapping array: {mapping_out_path.name}")

        # 3. Compile and serialize runtime index metadata profile
        meta_payload = {
            "embedding_model": self.expected_model,
            "embedding_dimension": self.embedding_dimension,
            "index_type": "IndexFlatIP",
            "metric": "cosine_similarity",
            "vectors": vector_count,
            "created_at": datetime.now(UTC).isoformat(),
            "version": "1.0.0"
        }
        with open(metadata_out_path, "w", encoding="utf-8") as meta_f:
            json.dump(meta_payload, meta_f, indent=4, ensure_ascii=False)
        logger.info(f"[+] Serialized Index Profile Metadata: {metadata_out_path.name}")