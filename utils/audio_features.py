"""
Audio Feature Extraction Module
===============================

This module extracts acoustic biomarkers from voice and cough audio files.
All calculations use established digital signal processing techniques.

Features extracted:
- Duration: Audio length in seconds
- Mean Pitch (F0): Fundamental frequency tracking
- RMS Energy: Amplitude/loudness over time
- Spectral Centroid: "Brightness" of sound
- Zero Crossing Rate: Voice vs noise indicator
- Jitter/Shimmer: Voice stability approximations

Author: Voice & Cough Mini-Screener Project
License: Educational use only - Non-diagnostic
"""

import librosa
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def load_audio(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa with standard sampling rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sampling rate (22050 Hz is standard for speech analysis)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Load audio file, resample to standard rate
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")


def extract_pitch_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract pitch-related features using librosa's fundamental frequency tracking.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
    
    Returns:
        Dictionary with pitch features
    """
    # Extract fundamental frequency using librosa's piptrack
    # fmin=50Hz, fmax=400Hz covers typical human voice range
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=400)
    
    # Get pitch values where magnitude is significant
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Only include valid pitch values
            pitch_values.append(pitch)
    
    if not pitch_values:
        return {
            'mean_pitch': 0.0,
            'pitch_std': 0.0,
            'jitter_approx': 0.0
        }
    
    pitch_values = np.array(pitch_values)
    
    # Calculate pitch statistics
    mean_pitch = np.mean(pitch_values)
    pitch_std = np.std(pitch_values)
    
    # Jitter approximation: coefficient of variation of pitch
    # Real jitter requires more complex algorithms, this is a simplified version
    jitter_approx = (pitch_std / mean_pitch) * 100 if mean_pitch > 0 else 0
    
    return {
        'mean_pitch': float(mean_pitch),
        'pitch_std': float(pitch_std),
        'jitter_approx': float(jitter_approx)
    }


def extract_energy_features(audio: np.ndarray) -> Dict[str, float]:
    """
    Extract energy-related features from audio signal.
    
    Args:
        audio: Audio signal array
    
    Returns:
        Dictionary with energy features
    """
    # RMS Energy: Root Mean Square amplitude
    rms_energy = np.sqrt(np.mean(audio**2))
    
    # Peak amplitude
    peak_amplitude = np.max(np.abs(audio))
    
    # Shimmer approximation: coefficient of variation of amplitude
    # Real shimmer requires more complex algorithms, this is simplified
    frame_length = 1024
    hop_length = 512
    
    # Calculate RMS for each frame
    rms_frames = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Shimmer approximation: variation in RMS energy
    shimmer_approx = (np.std(rms_frames) / np.mean(rms_frames)) * 100 if np.mean(rms_frames) > 0 else 0
    
    return {
        'rms_energy': float(rms_energy),
        'peak_amplitude': float(peak_amplitude),
        'shimmer_approx': float(shimmer_approx)
    }


def extract_spectral_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract spectral features from audio signal.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
    
    Returns:
        Dictionary with spectral features
    """
    # Spectral Centroid: "brightness" of sound (center of mass of spectrum)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    mean_spectral_centroid = np.mean(spectral_centroids)
    
    # Spectral Rolloff: Frequency below which 85% of energy lies
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    mean_spectral_rolloff = np.mean(spectral_rolloff)
    
    # Zero Crossing Rate: Rate of sign changes (voice vs noise indicator)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    mean_zcr = np.mean(zcr)
    
    # Spectral Bandwidth: Width of spectrum around centroid
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    mean_spectral_bandwidth = np.mean(spectral_bandwidth)
    
    return {
        'spectral_centroid': float(mean_spectral_centroid),
        'spectral_rolloff': float(mean_spectral_rolloff),
        'zero_crossing_rate': float(mean_zcr),
        'spectral_bandwidth': float(mean_spectral_bandwidth)
    }


def extract_all_features(file_path: str) -> pd.DataFrame:
    """
    Extract all acoustic features from an audio file.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Pandas DataFrame with extracted features
    """
    # Load audio
    audio, sr = load_audio(file_path)
    
    # Calculate duration
    duration = len(audio) / sr
    
    # Extract different feature groups
    pitch_features = extract_pitch_features(audio, sr)
    energy_features = extract_energy_features(audio)
    spectral_features = extract_spectral_features(audio, sr)
    
    # Combine all features
    all_features = {
        'duration': duration,
        **pitch_features,
        **energy_features,
        **spectral_features
    }
    
    # Create DataFrame with feature descriptions
    feature_df = pd.DataFrame([all_features])
    
    # Add human-readable descriptions
    feature_descriptions = {
        'duration': 'Duration (seconds)',
        'mean_pitch': 'Mean Pitch - F0 (Hz)',
        'pitch_std': 'Pitch Standard Deviation (Hz)',
        'jitter_approx': 'Jitter Approximation (%)',
        'rms_energy': 'RMS Energy',
        'peak_amplitude': 'Peak Amplitude',
        'shimmer_approx': 'Shimmer Approximation (%)',
        'spectral_centroid': 'Spectral Centroid (Hz)',
        'spectral_rolloff': 'Spectral Rolloff (Hz)',
        'zero_crossing_rate': 'Zero Crossing Rate',
        'spectral_bandwidth': 'Spectral Bandwidth (Hz)'
    }
    
    # Round numerical values for display
    for col in feature_df.columns:
        if col != 'duration':
            feature_df[col] = feature_df[col].round(3)
        else:
            feature_df[col] = feature_df[col].round(2)
    
    return feature_df, feature_descriptions


def calculate_risk_score(features: pd.DataFrame) -> float:
    """
    Calculate a more sophisticated analysis score based on extracted features.
    
    This score reflects acoustic irregularity patterns that might indicate
    different types of audio (cough vs voice vs breathing).
    
    Args:
        features: DataFrame with extracted features
    
    Returns:
        Analysis score between 0-100
    """
    # Extract key features for analysis
    jitter = features['jitter_approx'].iloc[0]
    shimmer = features['shimmer_approx'].iloc[0]
    pitch_std = features['pitch_std'].iloc[0]
    zcr = features['zero_crossing_rate'].iloc[0]
    rms_energy = features['rms_energy'].iloc[0]
    duration = features['duration'].iloc[0]
    spectral_centroid = features['spectral_centroid'].iloc[0]
    
    # More sophisticated scoring based on actual audio characteristics
    
    # 1. Voice stability (jitter/shimmer) - higher = more irregular
    voice_irregularity = min((jitter + shimmer) / 20.0, 1.0)  # Much more conservative
    
    # 2. Pitch variation - higher std = more pitch instability
    pitch_instability = min(pitch_std / 100.0, 1.0)  # More conservative
    
    # 3. Noise content (ZCR) - higher = more noise/breathiness
    noise_content = min(zcr * 20, 1.0)  # Scale down ZCR
    
    # 4. Energy characteristics - very low or very high energy is notable
    energy_score = 0
    if rms_energy < 0.005:  # Very quiet
        energy_score = 0.6
    elif rms_energy > 0.2:  # Very loud
        energy_score = 0.4
    else:
        energy_score = 0.1
    
    # 5. Spectral characteristics - coughs tend to have different spectral patterns
    spectral_score = 0
    if spectral_centroid < 500:  # Very low frequency content
        spectral_score = 0.5
    elif spectral_centroid > 4000:  # Very high frequency content
        spectral_score = 0.3
    else:
        spectral_score = 0.05
    
    # 6. Duration factor - very short or very long audio is notable
    duration_score = 0
    if duration < 0.5:  # Very short
        duration_score = 0.4
    elif duration > 10.0:  # Very long
        duration_score = 0.3
    else:
        duration_score = 0.05
    
    # Combine all factors with realistic weights - much more conservative
    analysis_score = (
        voice_irregularity * 0.30 +
        pitch_instability * 0.25 +
        noise_content * 0.20 +
        energy_score * 0.10 +
        spectral_score * 0.10 +
        duration_score * 0.05
    ) * 100
    
    # Ensure score is between 0-100
    return min(max(analysis_score, 0), 100)
