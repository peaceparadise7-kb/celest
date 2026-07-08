"""
Celest Machine Learning Subsystem - Audio Feature Service.
Handles numerical audio signal processing and statistical feature reduction using Librosa.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import librosa

logger = logging.getLogger("celest_audio_feature_service")


class AudioFeatureService:
    """
    Service layer responsible for loading digital audio bitstreams and computing
    standardized acoustic and timbral descriptors for recommendation models.
    """

    def __init__(self, hop_length: int = 512, n_mfcc: int = 20):
        """
        Initializes the feature extraction parameters.
        Native sample rate tracking is maintained by default via explicit constructor omission.
        """
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def extract_track_features(self, file_path: Path) -> Dict[str, Any]:
        """
        Loads an audio file and computes its mathematical feature profiles.
        Preserves the native audio sample rate by specifying sr=None.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Source file asset missing at: {file_path}")

        # Load audio bitstream preserving native sample rate, downmixing to mono natively
        y, sr = librosa.load(str(file_path), sr=None, mono=True)

        # Handle empty or silent audio streams safely
        if len(y) == 0:
            raise ValueError(f"Audio file contains no decodable sample payload: {file_path.name}")

        # 1. Mel-Frequency Cepstral Coefficients (MFCC) - 20 Coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        mfcc_std = np.std(mfccs, axis=1).tolist()

        # 2. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
        
        # 3. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
        
        # 5. Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)
        
        # 6. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)

        # 7. Global Tempo Estimation (BPM)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        tempo_values = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)
        tempo = float(tempo_values[0]) if len(tempo_values) > 0 else 0.0

        # 8. Chroma short-time Fourier transform (12-bin Pitch Profile)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        # Compile features into a clean, structured dictionary mapping layout
        return {
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spectral_rolloff_std": float(np.std(spectral_rolloff)),
            "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
            "zero_crossing_rate_std": float(np.std(zero_crossing_rate)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "tempo": tempo,
            "chroma": chroma_mean
        }