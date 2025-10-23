"""
Real Data Integration Module
============================

This module helps integrate real voice and cough datasets into the
Voice & Cough Mini-Screener for more accurate demonstrations.

Author: Voice & Cough Mini-Screener Project
License: Educational use only - Non-diagnostic
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import librosa
import soundfile as sf


def setup_data_directory(base_dir: str = "data") -> Dict[str, str]:
    """
    Create directory structure for real datasets.
    
    Args:
        base_dir: Base directory for all datasets
    
    Returns:
        Dictionary with dataset paths
    """
    datasets = {
        'coughvid': os.path.join(base_dir, 'coughvid'),
        'coswara': os.path.join(base_dir, 'coswara'),
        'covid_sounds': os.path.join(base_dir, 'covid_sounds'),
        'processed': os.path.join(base_dir, 'processed')
    }
    
    # Create directories
    for dataset_path in datasets.values():
        os.makedirs(dataset_path, exist_ok=True)
    
    return datasets


def process_coughvid_dataset(coughvid_path: str, output_path: str, max_files: int = 50) -> pd.DataFrame:
    """
    Process COUGHVID dataset for use in the app.
    
    Args:
        coughvid_path: Path to downloaded COUGHVID dataset
        output_path: Path to save processed files
        max_files: Maximum number of files to process (for demo)
    
    Returns:
        DataFrame with file metadata
    """
    print("Processing COUGHVID dataset...")
    
    # Find all WAV files
    wav_files = []
    for root, dirs, files in os.walk(coughvid_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    # Limit files for demo
    wav_files = wav_files[:max_files]
    
    metadata = []
    processed_count = 0
    
    for i, file_path in enumerate(wav_files):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=22050)
            
            # Skip very short or very long files
            duration = len(audio) / sr
            if duration < 0.5 or duration > 10:
                continue
            
            # Create processed filename
            filename = f"coughvid_{i:03d}.wav"
            output_file = os.path.join(output_path, filename)
            
            # Save processed file
            sf.write(output_file, audio, sr)
            
            # Extract basic features for metadata
            rms_energy = np.sqrt(np.mean(audio**2))
            
            metadata.append({
                'filename': filename,
                'original_path': file_path,
                'duration': duration,
                'rms_energy': rms_energy,
                'dataset': 'coughvid',
                'type': 'cough'
            })
            
            processed_count += 1
            print(f"Processed {processed_count}/{len(wav_files)} files...")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
    
    print(f"[OK] Processed {processed_count} files from COUGHVID dataset")
    return metadata_df


def process_coswara_dataset(coswara_path: str, output_path: str, max_files: int = 30) -> pd.DataFrame:
    """
    Process Coswara dataset for use in the app.
    
    Args:
        coswara_path: Path to downloaded Coswara dataset
        output_path: Path to save processed files
        max_files: Maximum number of files to process
    
    Returns:
        DataFrame with file metadata
    """
    print("Processing Coswara dataset...")
    
    # Find all WAV files
    wav_files = []
    for root, dirs, files in os.walk(coswara_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    wav_files = wav_files[:max_files]
    
    metadata = []
    processed_count = 0
    
    for i, file_path in enumerate(wav_files):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=22050)
            
            # Skip very short or very long files
            duration = len(audio) / sr
            if duration < 0.5 or duration > 10:
                continue
            
            # Determine file type from path
            file_type = 'unknown'
            if 'cough' in file_path.lower():
                file_type = 'cough'
            elif 'breath' in file_path.lower():
                file_type = 'breathing'
            elif 'voice' in file_path.lower() or 'speech' in file_path.lower():
                file_type = 'voice'
            
            # Create processed filename
            filename = f"coswara_{file_type}_{i:03d}.wav"
            output_file = os.path.join(output_path, filename)
            
            # Save processed file
            sf.write(output_file, audio, sr)
            
            # Extract basic features
            rms_energy = np.sqrt(np.mean(audio**2))
            
            metadata.append({
                'filename': filename,
                'original_path': file_path,
                'duration': duration,
                'rms_energy': rms_energy,
                'dataset': 'coswara',
                'type': file_type
            })
            
            processed_count += 1
            print(f"Processed {processed_count}/{len(wav_files)} files...")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
    
    print(f"[OK] Processed {processed_count} files from Coswara dataset")
    return metadata_df


def create_sample_selection(processed_path: str) -> List[str]:
    """
    Create a curated selection of the best samples for demo.
    
    Args:
        processed_path: Path to processed files
    
    Returns:
        List of selected filenames
    """
    metadata_file = os.path.join(processed_path, 'metadata.csv')
    
    if not os.path.exists(metadata_file):
        print("No metadata file found. Run dataset processing first.")
        return []
    
    # Load metadata
    df = pd.read_csv(metadata_file)
    
    # Select diverse samples
    selected_samples = []
    
    # Get samples by type
    for file_type in ['cough', 'voice', 'breathing']:
        type_samples = df[df['type'] == file_type]
        if len(type_samples) > 0:
            # Select sample with median duration
            median_duration = type_samples['duration'].median()
            closest_sample = type_samples.iloc[(type_samples['duration'] - median_duration).abs().argsort()[:1]]
            selected_samples.append(closest_sample['filename'].iloc[0])
    
    # Add a few more random samples
    remaining_samples = df[~df['filename'].isin(selected_samples)]
    if len(remaining_samples) > 0:
        random_samples = remaining_samples.sample(min(3, len(remaining_samples)))
        selected_samples.extend(random_samples['filename'].tolist())
    
    print(f"Selected {len(selected_samples)} samples for demo")
    return selected_samples


def setup_real_data():
    """
    Main function to set up real data integration.
    """
    print("Setting up real data integration...")
    
    # Create directory structure
    datasets = setup_data_directory()
    
    print("\n[DIR] Directory structure created:")
    for name, path in datasets.items():
        print(f"   {name}: {path}")
    
    print("\n[INFO] Next steps:")
    print("1. Download COUGHVID dataset from https://coughvid.epfl.ch/")
    print("2. Extract files to data/coughvid/ directory")
    print("3. Run: python utils/data_integration.py --process-coughvid")
    print("4. Update app.py to include real samples")
    
    return datasets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process real datasets for Voice & Cough Mini-Screener")
    parser.add_argument("--setup", action="store_true", help="Set up directory structure")
    parser.add_argument("--process-coughvid", action="store_true", help="Process COUGHVID dataset")
    parser.add_argument("--process-coswara", action="store_true", help="Process Coswara dataset")
    parser.add_argument("--coughvid-path", default="data/coughvid", help="Path to COUGHVID dataset")
    parser.add_argument("--coswara-path", default="data/coswara", help="Path to Coswara dataset")
    parser.add_argument("--max-files", type=int, default=50, help="Maximum files to process")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_real_data()
    elif args.process_coughvid:
        if os.path.exists(args.coughvid_path):
            process_coughvid_dataset(args.coughvid_path, "data/processed", args.max_files)
        else:
            print(f"[ERROR] COUGHVID dataset not found at {args.coughvid_path}")
            print("Please download it from https://coughvid.epfl.ch/ first")
    elif args.process_coswara:
        if os.path.exists(args.coswara_path):
            process_coswara_dataset(args.coswara_path, "data/processed", args.max_files)
        else:
            print(f"[ERROR] Coswara dataset not found at {args.coswara_path}")
            print("Please download it from https://coswara.iisc.ac.in/ first")
    else:
        print("Use --help to see available options")
