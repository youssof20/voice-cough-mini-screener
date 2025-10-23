#!/usr/bin/env python3
"""
Real Data Setup Script for Voice & Cough Mini-Screener
======================================================

This script helps you integrate real voice and cough datasets into your app.

Usage:
    python setup_real_data.py

Author: Voice & Cough Mini-Screener Project
License: Educational use only - Non-diagnostic
"""

import os
import sys
from utils.data_integration import setup_real_data, process_coughvid_dataset, process_coswara_dataset


def main():
    """Main setup function."""
    print("Voice & Cough Mini-Screener - Real Data Setup")
    print("=" * 50)
    
    # Set up directory structure
    print("\n1. Setting up directory structure...")
    datasets = setup_real_data()
    
    print("\n2. Checking for existing datasets...")
    
    # Check for COUGHVID
    coughvid_path = "data/coughvid"
    if os.path.exists(coughvid_path) and any(f.endswith('.wav') for f in os.listdir(coughvid_path)):
        print(f"[OK] Found COUGHVID dataset at {coughvid_path}")
        process_choice = input("Process COUGHVID dataset? (y/n): ").lower().strip()
        if process_choice == 'y':
            process_coughvid_dataset(coughvid_path, "data/processed", max_files=50)
    else:
        print(f"[MISSING] COUGHVID dataset not found at {coughvid_path}")
        print("   Download from: https://coughvid.epfl.ch/")
    
    # Check for Coswara
    coswara_path = "data/coswara"
    if os.path.exists(coswara_path) and any(f.endswith('.wav') for f in os.listdir(coswara_path)):
        print(f"[OK] Found Coswara dataset at {coswara_path}")
        process_choice = input("Process Coswara dataset? (y/n): ").lower().strip()
        if process_choice == 'y':
            process_coswara_dataset(coswara_path, "data/processed", max_files=30)
    else:
        print(f"[MISSING] Coswara dataset not found at {coswara_path}")
        print("   Download from: https://coswara.iisc.ac.in/")
    
    # Check if we have any processed data
    processed_path = "data/processed"
    if os.path.exists(processed_path):
        processed_files = [f for f in os.listdir(processed_path) if f.endswith('.wav')]
        if processed_files:
            print(f"\n[OK] Found {len(processed_files)} processed audio files")
            print("   Real samples are now available in the app!")
        else:
            print("\n[MISSING] No processed audio files found")
    else:
        print("\n[MISSING] No processed data directory found")
    
    print("\n" + "=" * 50)
    print("Next Steps:")
    print("1. Download datasets from the URLs above")
    print("2. Extract files to data/coughvid/ and data/coswara/")
    print("3. Run this script again to process the data")
    print("4. Start the app: streamlit run app.py")
    print("5. Real samples will appear in the sidebar!")
    
    print("\nRecommended Datasets:")
    print("   • COUGHVID: Best for cough analysis (20k+ files)")
    print("   • Coswara: Good for multiple sound types (1k+ files)")
    print("   • COVID-19 Sounds: Real-world mobile recordings")


if __name__ == "__main__":
    main()
