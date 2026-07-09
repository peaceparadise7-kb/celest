"""
Celest Machine Learning Subsystem - Hybrid Recommendation Service.
Exposes real-time online inference logic by blending semantic vector lookups and acoustic features.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import faiss
import numpy as np

from app.config.dataset_config import dataset_config

logger = logging.getLogger("celest_recommendation_service")


class RecommendationService:
    """
    Online Inference Service that delivers multi-modal hybrid music recommendations.
    Caches FAISS indices, row lookup lists, and track data profiles to maximize throughput.
    """

    # Production hybrid weighting constraints
    WEIGHT_SEMANTIC: float = 0.70
    WEIGHT_AUDIO: float = 0.30
    
    # Internal filter parameters for two-stage retrieval
    CANDIDATE_POOL_SIZE: int = 100

    def __init__(self) -> None:
        """
        Initializes the inference service and loads dependencies into local cache memory.
        """
        logger.info("Initializing Celest Hybrid Recommendation Engine Subsystem...")
        
        # Centralized resource caches
        self.index: Optional[faiss.IndexFlatIP] = None
        self.track_mapping: List[int] = []
        self.track_id_to_row: Dict[int, int] = {}
        self.manifest_lookup: Dict[int, Dict[str, Any]] = {}
        self.audio_feature_cache: Dict[int, np.ndarray] = {}
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        # Hydrate all data assets atomically at system initialization
        self._bootstrap_service_resources()

    def _bootstrap_service_resources(self) -> None:
        """Loads and caches FAISS indices, mapping arrays, and manifest data structures."""
        index_dir = dataset_config.dataset_vector_index_dir
        faiss_path = index_dir / "metadata_index.faiss"
        mapping_path = index_dir / "metadata_track_mapping.json"
        manifest_path = dataset_config.dataset_metadata_dir / "catalog_manifest.json"

        # 1. Populate Vector Index
        if not faiss_path.exists():
            raise FileNotFoundError(f"Critical System Fault: FAISS binary index file missing at: {faiss_path}")
        logger.info(f"Loading cached FAISS vector index: {faiss_path.name}")
        self.index = faiss.read_index(str(faiss_path.resolve()))
        logger.info(f"Loaded {self.index.ntotal} vectors inside the FAISS runtime memory map.")

        # 2. Populate Inverse Vector Identification Mapping Layers
        if not mapping_path.exists():
            raise FileNotFoundError(f"Critical System Fault: Track lookup registry missing at: {mapping_path}")
        with open(mapping_path, "r", encoding="utf-8") as map_f:
            self.track_mapping = json.load(map_f)
        
        # Build inverse dictionary index lookup for O(1) row location discovery
        self.track_id_to_row = {track_id: row_idx for row_idx, track_id in enumerate(self.track_mapping)}
        logger.info(f"Loaded {len(self.track_mapping)} zero-indexed row lookups from translation mapping.")

        # 3. Populate Fast Master Manifest Global Metadata Properties
        if not manifest_path.exists():
            raise FileNotFoundError(f"Critical System Fault: Base catalog manifest missing at: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as manifest_f:
            manifest_data = json.load(manifest_f)
            
        for track_node in manifest_data.get("tracks", []):
            t_id = track_node.get("track_id")
            if t_id is not None:
                self.manifest_lookup[int(t_id)] = track_node
        logger.info(f"Loaded {len(self.manifest_lookup)} master manifest entities into fast memory maps.")

        self._load_audio_feature_cache()

        logger.info(
            "RecommendationService initialized | "
            f"Vectors: {self.index.ntotal} | "
            f"Tracks: {len(self.track_mapping)} | "
            f"Cached Audio Features: {len(self.audio_feature_cache)}"
        )

    def _compile_feature_vector(self, feature_data: Dict[str, Any]) -> np.ndarray:
        """
        Extracts statistical properties from raw JSON structures and compiles them
        into a unified, standardized multi-dimensional feature array.
        """
        # Stacking features into a standardized configuration format
        feature_components = [
            feature_data.get("mfcc_mean", [0.0] * 20),
            feature_data.get("mfcc_std", [0.0] * 20),
            [feature_data.get("spectral_centroid_mean", 0.0)],
            [feature_data.get("spectral_centroid_std", 0.0)],
            [feature_data.get("spectral_bandwidth_mean", 0.0)],
            [feature_data.get("spectral_bandwidth_std", 0.0)],
            [feature_data.get("spectral_rolloff_mean", 0.0)],
            [feature_data.get("spectral_rolloff_std", 0.0)],
            [feature_data.get("zero_crossing_rate_mean", 0.0)],
            [feature_data.get("zero_crossing_rate_std", 0.0)],
            [feature_data.get("rms_mean", 0.0)],
            [feature_data.get("rms_std", 0.0)],
            [feature_data.get("tempo", 120.0)],
            feature_data.get("chroma", [0.0] * 12)
        ]

        # Flatten nested sequences cleanly into a single contiguous array row configuration
        flattened = []
        for block in feature_components:
            if isinstance(block, list):
                flattened.extend(block)
            else:
                flattened.append(float(block))

        return np.array(flattened, dtype=np.float32)
    
    def _load_audio_feature_cache(self) -> None:
        """
        Loads every processed audio feature into memory exactly once.
        This avoids repeated disk reads during recommendation requests.
        """
        logger.info("Caching processed audio features...")

        loaded = 0

        for track_id in self.manifest_lookup.keys():
            vector = self.load_audio_feature(track_id)

            if vector is not None:
                self.audio_feature_cache[track_id] = vector
                loaded += 1

        logger.info(
            f"Cached {loaded} audio feature vectors into memory."
        )

        if loaded > 0:
            all_vectors = np.stack(list(self.audio_feature_cache.values()))

            self.feature_mean = np.mean(all_vectors, axis=0)
            self.feature_std = np.std(all_vectors, axis=0)

            # Avoid division by zero for constant-valued features
            self.feature_std[self.feature_std == 0] = 1.0

            logger.info("Computed feature normalization statistics.")
        
    def load_audio_feature(self, track_id: int) -> Optional[np.ndarray]:
        """
        Loads the standalone audio feature JSON for a track and converts it to a NumPy vector.
        Bypasses failures gracefully if files are missing or corrupted.
        """
        track_node = self.manifest_lookup.get(track_id)
        if not track_node or not track_node.get("features_generated"):
            return None

        relative_path_str = track_node.get("relative_path", "")
        if not relative_path_str:
            return None

        # Resolve path matching audio features structure
        relative_path_obj = Path(relative_path_str)
        feature_json_path = (
            dataset_config.dataset_processed_audio_dir / relative_path_obj.parent / f"{relative_path_obj.stem}.json"
        )

        if not feature_json_path.exists():
            logger.debug(f"Acoustic features file absent for Track ID {track_id} at {feature_json_path.name}")
            return None

        try:
            with open(feature_json_path, "r", encoding="utf-8") as f:
                feature_data = json.load(f)
            return self._compile_feature_vector(feature_data)
        except Exception as err:
            logger.error(f"Failed to load or process feature configuration inside {feature_json_path.name}: {err}")
            return None

    def get_metadata_neighbors(self, track_id: int, pool_size: int) -> List[Tuple[int, float]]:
        """
        Queries the pre-cached FAISS index to find the nearest semantic neighbors using inner products.
        Returns a list of tuples containing (track_id, semantic_score).
        """
        row_idx = self.track_id_to_row.get(track_id)
        if row_idx is None:
            raise KeyError(f"Target Track ID '{track_id}' is not indexed inside the system vector map database.")

        # Reconstruct the vector directly from the internal FAISS index matrix layout
        query_vector = np.zeros((1, self.index.d), dtype=np.float32)
        query_vector[0] = self.index.reconstruct(row_idx)

        # Execute maximum inner product lookup (mathematically equivalent to cosine similarity since normalized)
        scores, indices = self.index.search(query_vector, pool_size)
        
        resolved_neighbors = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            neighbor_track_id = self.track_mapping[int(idx)]
            resolved_neighbors.append((neighbor_track_id, float(score)))

        return resolved_neighbors

    def calculate_audio_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Computes cosine similarity between two standardized audio feature vectors.
        """
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError(
                "Feature normalization statistics have not been initialized."
            )

        # Standardize both vectors using dataset-wide statistics
        vec_a = (vec_a - self.feature_mean) / self.feature_std
        vec_b = (vec_b - self.feature_mean) / self.feature_std

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cosine_val = np.dot(vec_a, vec_b) / (norm_a * norm_b)

        # Clamp for floating-point stability
        cosine_val = np.clip(cosine_val, -1.0, 1.0)

        # Convert [-1,1] → [0,1]
        return float((cosine_val + 1.0) / 2.0)

    def combine_scores(self, semantic_score: float, audio_score: float) -> float:
        """
        Computes a unified weighted average score across semantic and acoustic spaces.
        """
        return (semantic_score * self.WEIGHT_SEMANTIC) + (audio_score * self.WEIGHT_AUDIO)

    def recommend(self, track_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Computes multi-modal recommendations using a high-throughput, two-stage hybrid pipeline.
        
        Pipeline flow:
          1. Retrieve Stage 1 candidate pool using the fast FAISS semantic index lookup.
          2. Compute Stage 2 audio cosine similarity scores across the retrieved candidates.
          3. Calculate combined weighted scores, filter the target anchor, and sort descending.
        """
        logger.info(f"Processing recommendation request -> Query Track ID: {track_id} | Target Top-K: {top_k}")
        
        # Verify track existence before allocating pipeline execution metrics
        if track_id not in self.manifest_lookup:
            raise ValueError(f"Operational Error: Request Track ID {track_id} is absent from the master manifest.")

        # Stage 1: Fast Candidate Filtering via Semantic Index Lookup
        candidate_pool_size = max(top_k * 10, self.CANDIDATE_POOL_SIZE)

        candidate_pool = self.get_metadata_neighbors(
            track_id,
            candidate_pool_size
        )
        
        # Extract the source anchor vector array once to optimize calculation frames
        query_audio_vector = self.audio_feature_cache.get(track_id)
        if query_audio_vector is None:
            logger.warning(f"Audio descriptors absent for source anchor Track ID {track_id}. Audio similarity defaults to 0.5.")

        hybrid_candidates: List[Dict[str, Any]] = []

        # Stage 2: Target Selection and Multi-Modal Scoring
        for neighbor_id, semantic_score in candidate_pool:
            # Exclude the query track itself from recommendations
            if neighbor_id == track_id:
                continue

            # Load the corresponding candidate audio feature vector
            neighbor_audio_vector = self.audio_feature_cache.get(neighbor_id)
            
            if query_audio_vector is not None and neighbor_audio_vector is not None:
                # Compute standardized cosine similarity profile
                audio_score = self.calculate_audio_similarity(query_audio_vector, neighbor_audio_vector)
            else:
                # Fall back to a neutral 0.5 baseline score if feature assets are missing or corrupt
                audio_score = 0.5

            # Calculate unified hybrid metric allocation value
            hybrid_score = self.combine_scores(semantic_score, audio_score)

            track_meta = self.manifest_lookup.get(neighbor_id, {})

            hybrid_candidates.append({
                "track_id": neighbor_id,
                "title": track_meta.get("title", "Unknown"),
                "artist": track_meta.get("artist", "Unknown"),
                "album": track_meta.get("album", "Unknown"),
                "hybrid_score": round(hybrid_score, 6),
                "semantic_score": round(semantic_score, 6),
                "audio_score": round(audio_score, 6)
            })
            
        # Sort the hybrid candidates descending by score
        hybrid_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        logger.info(f"Hybrid ranking complete. Returning top {min(top_k, len(hybrid_candidates))} recommendations.")
        return hybrid_candidates[:top_k]