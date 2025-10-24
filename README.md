# Voice & Cough Mini-Screener

A web application that analyzes voice and cough audio files using digital signal processing techniques to extract acoustic features and visualize audio characteristics.

## Important Disclaimer

**This application is for EDUCATIONAL and DEMONSTRATION purposes only.**

- This is NOT a medical device and should NOT be used for diagnostic purposes
- The analysis scores are purely demonstrative and have no clinical validity
- Always consult healthcare professionals for medical concerns

## Features

- **Real Audio Analysis**: Extracts acoustic features from actual audio files
- **Feature Extraction**: Pitch, energy, spectral characteristics, voice stability metrics
- **Visualizations**: Waveforms, spectrograms, and pitch contours
- **Real Data**: Uses actual audio samples from the Coswara medical dataset
- **Interactive Interface**: Clean web interface built with Streamlit

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## Usage

1. **Upload an audio file** using the sidebar (WAV, MP3, M4A supported)
2. **Try real samples** - actual audio from the Coswara medical dataset
3. **View results** - acoustic features, visualizations, and analysis scores
4. **Explore features** - detailed explanations of each acoustic metric

## Technical Details

### Extracted Features
- **Duration**: Audio length in seconds
- **Mean Pitch (F0)**: Fundamental frequency tracking
- **RMS Energy**: Amplitude/loudness over time
- **Spectral Centroid**: Frequency content analysis
- **Zero Crossing Rate**: Voice vs noise indicator
- **Jitter/Shimmer**: Voice stability approximations

### Analysis Score
The analysis score (0-100) measures audio irregularity patterns:
- **Low (0-30)**: Very stable audio - smooth, consistent sound
- **Moderate (30-70)**: Some variation - normal speech patterns
- **High (70-100)**: Highly irregular - coughs, complex patterns

### Implementation
- **Audio Processing**: librosa for feature extraction
- **Visualization**: matplotlib for scientific plots
- **Web Interface**: Streamlit for interactive display
- **Data Source**: Coswara dataset (real medical audio)

## Project Structure

```
voice_cough_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── utils/
│   ├── audio_features.py     # Feature extraction algorithms
│   ├── visualizations.py     # Plotting functions
│   └── data_integration.py   # Dataset processing
└── data/
    ├── coswara/              # Original Coswara dataset
    └── processed/            # Processed audio samples
```

## Educational Value

This project demonstrates:
- Digital signal processing fundamentals
- Audio feature extraction techniques
- Biomedical signal analysis
- Web application development
- Real-world data processing

## Dataset Information

The application uses real audio samples from the **Coswara dataset**:
- **Source**: https://coswara.iisc.ac.in/
- **Content**: Real cough, breathing, and voice recordings
- **Format**: WAV files with clinical annotations
- **Usage**: Educational and research purposes only

## License

This project is for **educational use only**.
- Study, modify, and learn from the code
- Use for academic research projects
- Incorporate into educational materials
- No commercial or medical use
- No medical diagnosis or treatment

## Contributing

Contributions are welcome for:
- Feature improvements and new algorithms
- Visualization enhancements
- Documentation improvements
- Bug fixes and compatibility

## Support

For questions about this educational project:
- Check the code comments and documentation
- Review the feature explanations in the app
- Consult academic literature for research applications
- Always consult healthcare professionals for medical questions

---

**Remember: This tool demonstrates audio signal processing capabilities, not medical diagnosis. Always consult healthcare professionals for medical concerns.**