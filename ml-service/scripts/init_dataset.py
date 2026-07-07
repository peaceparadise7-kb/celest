from pathlib import Path

from app.config.dataset_config import dataset_config


def create_directory(path: Path):
    """
    Create a directory if it doesn't already exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {path}")


def initialize_dataset():
    print("\n🚀 Initializing Celest Dataset Structure...\n")

    directories = [
        dataset_config.dataset_base_dir,
        dataset_config.dataset_raw_audio_dir,
        dataset_config.dataset_raw_lyrics_dir,
        dataset_config.dataset_metadata_dir,
        dataset_config.dataset_processed_audio_dir,
        dataset_config.dataset_processed_text_dir,
        dataset_config.dataset_cache_dir,
        dataset_config.dataset_exports_dir,
        dataset_config.dataset_models_dir,
    ]

    for directory in directories:
        create_directory(directory)

    print("\n🎉 Dataset folder structure initialized successfully.")


if __name__ == "__main__":
    initialize_dataset()