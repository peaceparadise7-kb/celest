from app.config.dataset_config import dataset_config


def verify():
    print("\n🔍 Verifying Dataset Structure...\n")

    paths = [
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

    success = True

    for path in paths:
        if path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ Missing: {path}")
            success = False

    if success:
        print("\n🎉 Dataset verification successful.")
    else:
        print("\n⚠️ Dataset verification failed.")


if __name__ == "__main__":
    verify()