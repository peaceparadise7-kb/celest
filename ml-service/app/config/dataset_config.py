from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetSettings(BaseSettings):
    """
    Centralized Dataset Configuration.

    All dataset locations are read from the .env file.
    Every future ML script should import this object instead of
    hardcoding filesystem paths.
    """

    dataset_base_dir: Path = Path("../datasets")
    dataset_raw_audio_dir: Path = Path("../datasets/raw/audio")
    dataset_raw_lyrics_dir: Path = Path("../datasets/raw/lyrics")
    dataset_metadata_dir: Path = Path("../datasets/metadata")

    dataset_processed_audio_dir: Path = Path("../datasets/processed/audio_features")
    dataset_processed_metadata_dir: Path = Path("../datasets/processed/metadata_embeddings")
    dataset_vector_index_dir: Path = Path("../datasets/processed/vector_indices")

    dataset_cache_dir: Path = Path("../datasets/cache")
    dataset_exports_dir: Path = Path("../datasets/exports")
    dataset_models_dir: Path = Path("../datasets/models")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


dataset_config = DatasetSettings()