"""
Visualization Module
====================

This module creates scientific visualizations for audio analysis including
waveforms, spectrograms, and feature displays using matplotlib.

Author: Voice & Cough Mini-Screener Project
License: Educational use only - Non-diagnostic
"""

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from typing import Tuple
import io
import base64


def create_waveform_plot(audio: np.ndarray, sr: int, title: str = "Audio Waveform") -> plt.Figure:
    """
    Create a waveform visualization showing amplitude over time.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    # Create time axis
    time = np.linspace(0, len(audio) / sr, len(audio))
    
    # Create figure with scientific styling
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot waveform
    ax.plot(time, audio, linewidth=0.8, color='#2E86AB')
    ax.fill_between(time, audio, alpha=0.3, color='#2E86AB')
    
    # Styling
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable limits
    ax.set_xlim(0, len(audio) / sr)
    ax.set_ylim(-1.1, 1.1)
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    return fig


def create_spectrogram_plot(audio: np.ndarray, sr: int, title: str = "Spectrogram") -> plt.Figure:
    """
    Create a spectrogram visualization showing frequency content over time.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    # Create figure with scientific styling
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute spectrogram using librosa
    # n_fft=2048 gives good frequency resolution
    # hop_length=512 gives good time resolution
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048, hop_length=512)), ref=np.max)
    
    # Display spectrogram
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, 
                                   hop_length=512, ax=ax, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)', fontsize=12)
    
    # Styling
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set frequency range to focus on human voice (0-4000 Hz)
    ax.set_ylim(0, 4000)
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    return fig


def create_pitch_curve_plot(audio: np.ndarray, sr: int, title: str = "Pitch Contour") -> plt.Figure:
    """
    Create a pitch contour visualization showing fundamental frequency over time.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    # Extract pitch using librosa
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=400)
    
    # Get pitch values where magnitude is significant
    pitch_values = []
    times = []
    hop_length = 512
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Only include valid pitch values
            pitch_values.append(pitch)
            times.append(t * hop_length / sr)
    
    if not pitch_values:
        # If no pitch detected, create empty plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'No pitch detected in audio', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Pitch (Hz)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot pitch contour
    ax.plot(times, pitch_values, linewidth=2, color='#E63946', marker='o', markersize=3)
    
    # Styling
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Pitch (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable limits
    if pitch_values:
        ax.set_ylim(max(0, min(pitch_values) - 20), max(pitch_values) + 20)
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    return fig


def create_combined_plot(audio: np.ndarray, sr: int, title: str = "Audio Analysis") -> plt.Figure:
    """
    Create a combined plot showing waveform, spectrogram, and pitch contour.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        title: Overall plot title
    
    Returns:
        Matplotlib figure object
    """
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Waveform
    time = np.linspace(0, len(audio) / sr, len(audio))
    axes[0].plot(time, audio, linewidth=0.8, color='#2E86AB')
    axes[0].fill_between(time, audio, alpha=0.3, color='#2E86AB')
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].set_title('Waveform', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, len(audio) / sr)
    axes[0].set_ylim(-1.1, 1.1)
    
    # 2. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048, hop_length=512)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, 
                                   hop_length=512, ax=axes[1], cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=10)
    axes[1].set_title('Spectrogram', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 4000)
    
    # 3. Pitch contour
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=400)
    pitch_values = []
    times = []
    hop_length = 512
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
            times.append(t * hop_length / sr)
    
    if pitch_values:
        axes[2].plot(times, pitch_values, linewidth=2, color='#E63946', marker='o', markersize=2)
        axes[2].set_ylim(max(0, min(pitch_values) - 20), max(pitch_values) + 20)
    else:
        axes[2].text(0.5, 0.5, 'No pitch detected', ha='center', va='center', 
                    transform=axes[2].transAxes, fontsize=10)
    
    axes[2].set_xlabel('Time (seconds)', fontsize=10)
    axes[2].set_ylabel('Pitch (Hz)', fontsize=10)
    axes[2].set_title('Pitch Contour', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert matplotlib figure to base64 string for Streamlit display.
    
    Args:
        fig: Matplotlib figure object
    
    Returns:
        Base64 encoded string
    """
    # Save figure to bytes
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    
    return image_base64


def create_feature_summary_plot(features: dict, title: str = "Feature Summary") -> plt.Figure:
    """
    Create a bar chart showing key acoustic features.
    
    Args:
        features: Dictionary of extracted features
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    # Select key features for visualization
    key_features = {
        'Mean Pitch (Hz)': features.get('mean_pitch', 0),
        'RMS Energy': features.get('rms_energy', 0),
        'Spectral Centroid (Hz)': features.get('spectral_centroid', 0),
        'Jitter (%)': features.get('jitter_approx', 0),
        'Shimmer (%)': features.get('shimmer_approx', 0)
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(range(len(key_features)), list(key_features.values()), 
                  color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'])
    
    # Styling
    ax.set_xlabel('Acoustic Features', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(key_features)))
    ax.set_xticklabels(list(key_features.keys()), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, key_features.values())):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(key_features.values())*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    return fig
