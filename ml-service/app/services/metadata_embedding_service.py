"""
Celest Machine Learning Subsystem - Metadata Embedding Service.
Handles structural text normalization, relational genre lookup, and vector generation.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("celest_metadata_embedding_service")


class MetadataEmbeddingService:
    """
    Service layer responsible for mapping track metadata fields into semantic text
    documents and converting them into dense vector embeddings.
    """

    def __init__(self, metadata_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the semantic text transformer service and pre-caches the genre index map.
        """
        self.model_name = model_name
        logger.info(f"Loading sentence transformer model target: {self.model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Load the genre conversion index map exactly once during initialization
        self.genre_map = self._load_genre_map(metadata_dir)

    def _load_genre_map(self, metadata_dir: Path) -> Dict[int, str]:
        """
        Parses genres.csv into an in-memory lookup table to resolve numerical genre IDs.
        """
        genre_map: Dict[int, str] = {}
        genres_csv_path = metadata_dir / "genres.csv"

        if not genres_csv_path.exists():
            logger.warning(f"genres.csv missing at {genres_csv_path}. Numerical IDs will not resolve.")
            return genre_map

        try:
            with open(genres_csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                # Identify explicit index locations for structural safety
                genre_id_idx = 0
                title_idx = 3 # Default standard FMA position for 'title' column in genres.csv
                
                if header:
                    try:
                        genre_id_idx = header.index("genre_id")
                        title_idx = header.index("title")
                    except ValueError:
                        pass # Maintain standard fallback constraints if columns differ

                for row in reader:
                    if not row or len(row) <= max(genre_id_idx, title_idx):
                        continue
                    try:
                        g_id = int(row[genre_id_idx].strip())
                        g_title = row[title_idx].strip()
                        if g_title:
                            genre_map[g_id] = g_title
                    except ValueError:
                        continue
                        
            logger.info(f"Loaded {len(genre_map)} genre mappings from genres.csv.")
        except Exception as err:
            logger.error(f"Failed to compile genre map index: {err}")

        return genre_map

    def resolve_genre_names(self, genre_ids: Optional[List[Any]]) -> str:
        """
        Converts a list of numerical genre IDs or string tokens into a comma-separated string.
        """
        if not genre_ids or not isinstance(genre_ids, list):
            return "Unknown"

        resolved_titles: List[str] = []
        for raw_id in genre_ids:
            try:
                clean_id = int(float(str(raw_id).strip()))
                if clean_id in self.genre_map:
                    resolved_titles.append(self.genre_map[clean_id])
            except (ValueError, TypeError):
                # Fall back to raw string token if conversion fails
                if str(raw_id).strip():
                    resolved_titles.append(str(raw_id).strip())

        return ", ".join(resolved_titles) if resolved_titles else "Unknown"

    def extract_year(self, release_info: Optional[Any]) -> str:
        """
        Extracts a clean 4-digit calendar year string from raw release descriptions.
        """
        if not release_info or not str(release_info).strip():
            return "Unknown"
        
        match = re.search(r"\b(19\d\d|20\d\d)\b", str(release_info))
        return match.group(1) if match else "Unknown"

    def _normalize_string(self, value: Optional[Any]) -> str:
        """
        Helper method to strictly evaluate values, converting empty strings 
        or whitespace-only fields into 'Unknown'.
        """
        if value is None:
            return "Unknown"
        clean_str = str(value).strip()
        return clean_str if clean_str else "Unknown"

    def build_metadata_document(self, track_meta: Dict[str, Any]) -> str:
        """
        Transforms relational track properties into an enriched natural language document.
        Missing values or whitespace-only records are normalized to 'Unknown'.
        """
        title = self._normalize_string(track_meta.get("title"))
        artist = self._normalize_string(track_meta.get("artist"))
        album = self._normalize_string(track_meta.get("album"))
        track_num = self._normalize_string(track_meta.get("track_number"))
        
        # Resolve duration modeling text context safely
        raw_duration = track_meta.get("duration")
        if raw_duration is not None and str(raw_duration).strip():
            try:
                duration_str = f"{int(round(float(raw_duration)))} seconds"
            except (ValueError, TypeError):
                duration_str = "Unknown"
        else:
            duration_str = "Unknown"
        
        # Resolve text names for genre arrays
        genres_str = self.resolve_genre_names(track_meta.get("genres"))
        
        # Extract calendar year sequences safely
        year_str = self.extract_year(track_meta.get("release_information"))

        # Construct an enriched, human-readable semantic context document
        return (
            f"Song Title: {title}\n"
            f"Artist: {artist}\n"
            f"Album: {album}\n"
            f"Genres: {genres_str}\n"
            f"Released: {year_str}\n"
            f"Duration: {duration_str}\n"
            f"Track Number: {track_num}"
        )

    def generate_embedding(self, text_document: str) -> List[float]:
        """
        Converts a normalized text document into a normalized dense 384-dimensional float32 vector.
        Uses torch.no_grad() to skip tracking gradients.
        """
        with torch.no_grad():
            vector = self.model.encode(
                text_document, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
        return vector.astype(np.float32).tolist()