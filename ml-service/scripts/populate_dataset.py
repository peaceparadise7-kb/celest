import os
import sys
import csv
import json
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Any, Set, Tuple, Optional

from app.config.dataset_config import dataset_config

# Configure structured, production-ready logging outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("celest_dataset_population")

# Strict asset extensions allowed within processing bounds
SUPPORTED_FORMATS: Set[str] = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def generate_deterministic_id(absolute_path_str: str) -> str:
    """
    Creates a unique identifier for a track using a SHA-256 hash.
    Ensures structural keys remain invariant across execution lifetimes.
    """
    sha_handle = hashlib.sha256(absolute_path_str.encode("utf-8"))
    return f"fma---{sha_handle.hexdigest()[:24]}"


def extract_fma_track_id(filename: str) -> Optional[int]:
    """
    Extracts the numerical track index out of an FMA filename string.
    Example: '000002.mp3' -> 2
    """
    stem_name = Path(filename).stem
    try:
        return int(stem_name)
    except ValueError:
        logger.warning(f"Could not parse numerical FMA Track ID from string: {filename}")
        return None


def parse_technical_audio_metadata(file_path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    """
    Uses mutagen to read structural audio layer metadata directly from bitstream boundaries.
    Returns: (duration_seconds, sample_rate_hz, bitrate_bps, channels)
    """
    duration, sample_rate, bitrate, channels = None, None, None, None
    try:
        from mutagen import File as MutagenFile
        audio_stream = MutagenFile(file_path, easy=True)
        if audio_stream is not None and audio_stream.info:
            info = audio_stream.info
            duration = round(float(info.length), 2) if hasattr(info, "length") else None
            sample_rate = int(info.sample_rate) if hasattr(info, "sample_rate") else None
            bitrate = int(info.bitrate) if hasattr(info, "bitrate") else None
            channels = int(info.channels) if hasattr(info, "channels") else None
    except Exception as err:
        logger.debug(f"Technical audio metadata parsing bypassed for {file_path.name}: {err}")
    
    return duration, sample_rate, bitrate, channels


def resolve_fma_column_mappings(header_rows: list[list[str]]) -> Dict[str, int]:
    """
    Parses FMA's unique 3-row hierarchical header block to securely isolate column index bindings.
    """
    mappings = {"track_id": 0} # Row index 0 is always the tracking structural track_id column
    
    # Extract distinct row tiers safely
    row_0 = header_rows[0] # Aggregation target category tier (track, artist, album)
    row_1 = header_rows[1] # Metric data attribute keys (title, tags, ID)
    
    for idx in range(min(len(row_0), len(row_1))):
        cat = row_0[idx].strip().lower()
        attr = row_1[idx].strip().lower()
        
        if cat == "track" and attr == "title":
            mappings["title"] = idx
        elif cat == "artist" and attr == "name":
            mappings["artist"] = idx
        elif cat == "album" and attr == "title":
            mappings["album"] = idx
        elif cat == "track" and attr == "genres":
            mappings["genres"] = idx
        elif cat == "track" and attr == "number":
            mappings["track_number"] = idx
        elif cat == "album" and attr == "date_released":
            mappings["release_info"] = idx

    # Enforce safe structural defaults if column shapes deviate from standard FMA structures
    fallback_defaults = {
        "title": 52, "artist": 26, "album": 5, "genres": 41, "track_number": 51, "release_info": 12
    }
    for key, default_idx in fallback_defaults.items():
        if key not in mappings:
            logger.warning(f"Metadata header match absent for field key '{key}'. Binding default index reference: {default_idx}")
            mappings[key] = default_idx
            
    return mappings


def load_authoritative_metadata(metadata_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    Streams tracks.csv line by line to compile a metadata lookup cache.
    Dynamically tracks internal field indices to decouple mapping rules from static schemas.
    """
    tracks_csv_path = metadata_dir / "tracks.csv"
    lookup_index: Dict[int, Dict[str, Any]] = {}
    
    if not tracks_csv_path.exists():
        logger.error(f"Critical CSV missing. Authoritative file not found at: {tracks_csv_path}")
        return lookup_index

    logger.info(f"Opening authoritative metadata source: {tracks_csv_path.name}")
    try:
        with open(tracks_csv_path, mode="r", encoding="utf-8", errors="ignore") as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Read and capture structural multi-tier layout lines (Rows 0, 1, 2)
            header_rows = [next(csv_reader, None) for _ in range(3)]
            if any(h is None for h in header_rows):
                logger.error("CSV formatting error detected: Multi-tier row indices are incomplete.")
                return lookup_index
                
            column_idx = resolve_fma_column_mappings(header_rows)
            
            for row in csv_reader:
                if not row or len(row) < max(column_idx.values()):
                    continue
                try:
                    raw_id = row[column_idx["track_id"]].strip()
                    if not raw_id:
                        continue
                    
                    track_id = int(raw_id)
                    
                    # Parse genre arrays safely if structured correctly
                    genres_raw = row[column_idx["genres"]].strip()
                    genres_list = []
                    if genres_raw.startswith("[") and genres_raw.endswith("]"):
                        try:
                            genres_list = json.loads(genres_raw.replace("'", '"'))
                        except Exception:
                            genres_list = [g.strip() for g in genres_raw[1:-1].split(",") if g.strip()]

                    lookup_index[track_id] = {
                        "title": row[column_idx["title"]].strip() if row[column_idx["title"]] else None,
                        "artist": row[column_idx["artist"]].strip() if row[column_idx["artist"]] else None,
                        "album": row[column_idx["album"]].strip() if row[column_idx["album"]] else None,
                        "genres": genres_list,
                        "track_number": int(row[column_idx["track_number"]]) if row[column_idx["track_number"]].strip().isdigit() else None,
                        "release_info": row[column_idx["release_info"]].strip() if row[column_idx["release_info"]] else None
                    }
                except (ValueError, IndexError):
                    continue
                    
        logger.info(f"Successfully cached metadata configurations for {len(lookup_index)} tracks.")
    except Exception as global_err:
        logger.error(f"Failed to stream metadata text vectors: {global_err}")
        
    return lookup_index


def run_pipeline_population() -> None:
    """Executes the dataset search sweep and populates the master catalog manifest."""
    logger.info("Initializing Celest Dataset Population Layer...")
    
    audio_search_root = dataset_config.dataset_raw_audio_dir
    metadata_source_dir = dataset_config.dataset_metadata_dir
    manifest_ledger_file = dataset_config.dataset_metadata_dir / "catalog_manifest.json"

    if not manifest_ledger_file.exists():
        logger.error(f"Target catalog tracking point missing on disk: {manifest_ledger_file}")
        sys.exit(1)

    try:
        with open(manifest_ledger_file, "r", encoding="utf-8") as file_read:
            manifest_state = json.load(file_read)
    except Exception as file_err:
        logger.error(f"Failed to parse base system manifest format: {file_err}")
        sys.exit(1)

    manifest_state.setdefault("tracks", [])
    
    existing_paths_index: Set[str] = {
        track["absolute_path"] for track in manifest_state["tracks"] if "absolute_path" in track
    }

    fma_metadata_cache = load_authoritative_metadata(metadata_source_dir)
    run_summary = {"discovered": 0, "registered": 0, "duplicates": 0, "failures": 0}
    
    logger.info(f"Scanning target raw tracking directory: {audio_search_root.resolve()}")
    
    for path_root, _, catalog_files in os.walk(audio_search_root):
        for tracking_file in catalog_files:
            file_disk_path = Path(path_root) / tracking_file
            file_extension = file_disk_path.suffix.lower()
            
            if file_extension not in SUPPORTED_FORMATS:
                continue
                
            run_summary["discovered"] += 1
            absolute_uri_string = str(file_disk_path.resolve())
            
            if absolute_uri_string in existing_paths_index:
                run_summary["duplicates"] += 1
                continue
                
            track_numeric_id = extract_fma_track_id(tracking_file)
            if track_numeric_id is None:
                logger.error(f"[!] Processing structural error. Track ID missing from naming: {tracking_file}")
                run_summary["failures"] += 1
                continue
                
            try:
                duration, sampling_rate, nominal_bitrate, audio_channels = parse_technical_audio_metadata(file_disk_path)
                file_byte_size = file_disk_path.stat().st_size
                cryptographic_id = generate_deterministic_id(absolute_uri_string)
                
                db_profile = fma_metadata_cache.get(
                    track_numeric_id, 
                    {"title": None, "artist": None, "album": None, "genres": [], "track_number": None, "release_info": None}
                )
                
                manifest_record: Dict[str, Any] = {
                    "id": cryptographic_id,
                    "track_id": track_numeric_id,
                    "filename": tracking_file,
                    "absolute_path": absolute_uri_string,
                    "relative_path": str(file_disk_path.relative_to(audio_search_root)),
                    "extension": file_extension,
                    "filesize": file_byte_size,
                    "duration": duration,
                    "sample_rate": sampling_rate,
                    "bitrate": nominal_bitrate,
                    "channels": audio_channels,
                    "title": db_profile["title"],
                    "artist": db_profile["artist"],
                    "album": db_profile["album"],
                    "genres": db_profile["genres"],
                    "track_number": db_profile["track_number"],
                    "release_information": db_profile["release_info"],
                    "processed": False,
                    "features_generated": False,
                    "embeddings_generated": False,
                    "created_timestamp": datetime.now(UTC).isoformat()
                }
                
                manifest_state["tracks"].append(manifest_record)
                existing_paths_index.add(absolute_uri_string)
                run_summary["registered"] += 1
                
                # Periodic verification output pass (every 100 successful entries)
                if run_summary["registered"] % 100 == 0:
                    logger.info(f"[*] Pipeline progress verification checkpoint: Registered {run_summary['registered']} tracks successfully.")
                
            except Exception as active_processing_fault:
                logger.error(f"[!] Unexpected error processing track {tracking_file}: {active_processing_fault}")
                run_summary["failures"] += 1

    try:
        with open(manifest_ledger_file, "w", encoding="utf-8") as file_write_back:
            json.dump(manifest_state, file_write_back, indent=4, ensure_ascii=False)
    except Exception as fatal_write_error:
        logger.error(f"Critical error writing back manifest mutations: {fatal_write_error}")
        sys.exit(1)

    print("\n" + "="*50)
    print("       DATASET POPULATION RUNTIME SUMMARY")
    print("="*50)
    print(f" Tracks Discovered:    {run_summary['discovered']}")
    print(f" Tracks Registered:    {run_summary['registered']}")
    print(f" Duplicates Skipped:   {run_summary['duplicates']}")
    print(f" Processing Failures:  {run_summary['failures']}")
    print("="*50)
    logger.info("🎉 Task 5.5 Execution Suite complete. Manifest file finalized.")


if __name__ == "__main__":
    run_pipeline_population()