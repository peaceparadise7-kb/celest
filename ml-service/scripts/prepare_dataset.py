import json
from datetime import datetime, UTC
from pathlib import Path

from app.config.dataset_config import dataset_config


def create_manifest():
    manifest = {
        "manifest_version": "1.0.0",
        "creation_timestamp": datetime.now(UTC).isoformat(),
        "dataset_source": "pending",
        "preprocessing_version": "pending",
        "text_embedding_model": None,
        "tracks": []
    }

    manifest_path = (
        dataset_config.dataset_metadata_dir / "catalog_manifest.json"
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ Created manifest: {manifest_path}")


if __name__ == "__main__":
    create_manifest()