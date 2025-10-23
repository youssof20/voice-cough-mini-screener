# 🎤 Voice & Cough Mini-Screener

A complete demonstration application that analyzes voice and cough audio files using **digital signal processing** techniques to extract acoustic biomarkers and visualize audio characteristics. This project demonstrates real-world audio analysis capabilities using established DSP methods.

## ⚠️ IMPORTANT DISCLAIMER

**This application is for EDUCATIONAL and DEMONSTRATION purposes only.**

- ❌ **NOT a medical device** - Do not use for diagnostic purposes
- ❌ **NOT clinically validated** - Risk scores are purely demonstrative  
- ❌ **NOT FDA approved** - No medical claims or recommendations
- ✅ **Educational tool** - Learn audio signal processing and biomarker extraction
- ✅ **Research demo** - Explore digital health possibilities

**Always consult healthcare professionals for medical concerns.**

## 🎯 Project Overview

This project demonstrates how **digital signal processing** can extract meaningful acoustic features from voice and cough sounds using established DSP algorithms. It's designed as a learning tool for:

- **Students** learning audio signal processing and DSP fundamentals
- **Researchers** exploring biomedical signal analysis applications  
- **Developers** building audio analysis tools with real data
- **Engineers** understanding voice/cough analysis techniques

### Key Features

- **Real Feature Extraction**: Pitch, energy, spectral, and stability metrics using DSP
- **Scientific Visualizations**: Waveforms, spectrograms, and pitch contours
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Real Medical Data**: Actual audio samples from Coswara dataset
- **Educational Content**: Detailed explanations of DSP techniques

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Process real data** (if you have Coswara dataset):
   ```bash
   python utils/data_integration.py --process-coswara
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### First Time Setup

The application uses **real audio samples** from the Coswara medical dataset. You can also upload your own audio files (WAV, MP3, M4A formats supported).

## 📊 Features Extracted

### Pitch Features
- **Mean Pitch (F0)**: Fundamental frequency of voice (Hz)
- **Pitch Standard Deviation**: Variation in pitch over time
- **Jitter Approximation**: Micro-variations in pitch (voice stability indicator)

### Energy Features  
- **RMS Energy**: Root mean square amplitude (loudness)
- **Peak Amplitude**: Maximum volume level
- **Shimmer Approximation**: Micro-variations in amplitude

### Spectral Features
- **Spectral Centroid**: "Brightness" of sound (center of mass of spectrum)
- **Spectral Rolloff**: Frequency below which 85% of energy lies
- **Zero Crossing Rate**: Rate of sign changes (voice vs noise indicator)
- **Spectral Bandwidth**: Width of spectrum around centroid

## 🔬 Technical Implementation

### Audio Processing Pipeline

1. **Audio Loading**: Uses librosa for robust audio file handling
2. **Preprocessing**: Resampling to 22,050 Hz (standard for speech analysis)
3. **Feature Extraction**: Frame-based analysis with 1024-sample windows
4. **Pitch Tracking**: librosa's piptrack algorithm (50-400 Hz range)
5. **Spectral Analysis**: 2048-point FFT with 512-sample hop length
6. **Visualization**: matplotlib for scientific-quality plots

### Risk Score Calculation

The "Health Risk Indicator" is a **demonstration metric only** that combines:
- Jitter and shimmer variations (voice stability)
- Pitch variability (neurological indicators)  
- Energy levels (respiratory strength)
- Spectral characteristics (voice quality)

**This score has no clinical validity and is purely educational.**

## 📁 Project Structure

```
voice_cough_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── utils/
│   ├── __init__.py
│   ├── audio_features.py     # Feature extraction algorithms
│   ├── visualizations.py    # Plotting and visualization functions
│   └── sample_generator.py  # Demo audio sample creation
└── samples/
    ├── demo_voice.wav        # Generated voice sample
    ├── demo_cough.wav        # Generated cough sample
    └── demo_breathy_voice.wav # Generated breathy voice sample
```

## 🎵 Demo Samples

The application includes three pre-generated synthetic samples:

1. **Normal Voice** (`demo_voice.wav`): Simulated speech with harmonics
2. **Cough Sample** (`demo_cough.wav`): Simulated cough with frequency sweeps
3. **Breathy Voice** (`demo_breathy_voice.wav`): Simulated breathy speech

These samples are programmatically generated and contain no copyrighted material.

### 🎯 Adding Real Data (Recommended)

For more accurate demonstrations, integrate real voice and cough datasets:

#### Quick Setup:
```bash
# 1. Set up data directories
python setup_real_data.py

# 2. Download COUGHVID dataset (recommended)
# Visit: https://coughvid.epfl.ch/
# Extract to: data/coughvid/

# 3. Process the dataset
python utils/data_integration.py --process-coughvid

# 4. Start the app - real samples will appear in sidebar
streamlit run app.py
```

#### Available Datasets:
- **[COUGHVID](https://coughvid.epfl.ch/)**: 20,000+ real cough recordings (best option)
- **[Coswara](https://coswara.iisc.ac.in/)**: 1,000+ respiratory sounds (multiple types)
- **[COVID-19 Sounds](https://covid-19-sounds.org/)**: Mobile app recordings

## 📚 Open Datasets for Real Testing

For testing with real audio data, consider these open datasets:

### Voice & Speech Datasets
- **[COUGHVID](https://coughvid.epfl.ch/)**: COVID-19 cough detection dataset
- **[Coswara](https://coswara.iisc.ac.in/)**: COVID-19 sounds dataset  
- **[COVID-19 Sounds](https://covid-19-sounds.org/)**: Respiratory sounds collection
- **[Common Voice](https://commonvoice.mozilla.org/)**: Multilingual speech dataset
- **[LibriSpeech](https://www.openslr.org/12/)**: Large-scale speech recognition corpus

### Respiratory Sound Datasets
- **[ICBHI](https://bhichallenge.med.auth.gr/)**: Lung sound database
- **[Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)**: Breath and cough sounds

## 🔧 Customization

### Adding New Features

To add new acoustic features, modify `utils/audio_features.py`:

```python
def extract_custom_feature(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract your custom feature here."""
    # Your feature extraction code
    return {'custom_feature': value}
```

### Modifying Visualizations

To add new plots, modify `utils/visualizations.py`:

```python
def create_custom_plot(audio: np.ndarray, sr: int) -> plt.Figure:
    """Create your custom visualization."""
    # Your plotting code
    return fig
```

### Changing Risk Score Algorithm

To modify the risk calculation, edit the `calculate_risk_score()` function in `utils/audio_features.py`.

## 🎓 Educational Value

This project teaches:

### Digital Signal Processing
- Audio file handling and preprocessing
- Fourier transforms and spectral analysis
- Pitch tracking algorithms
- Frame-based processing techniques

### Biomedical Signal Analysis
- Acoustic biomarker extraction
- Voice quality assessment metrics
- Respiratory sound analysis
- Clinical feature interpretation

### Web Application Development
- Streamlit framework usage
- Interactive data visualization
- File upload and processing
- Responsive web design

### Python Programming
- Object-oriented design
- Scientific computing with NumPy/SciPy
- Data visualization with matplotlib/plotly
- Audio processing with librosa

## 🚨 Ethical Considerations

### Privacy & Data Protection
- No audio data is stored or transmitted
- All processing happens locally
- Uploaded files are temporarily cached only
- Demo samples contain no personal information

### Medical Ethics
- Clear disclaimers about non-diagnostic use
- Educational purpose explicitly stated
- No medical claims or recommendations
- Encourages professional medical consultation

### Responsible Development
- Open source for transparency
- Educational focus over commercial use
- Emphasis on learning and research
- No misleading medical implications

## 🤝 Contributing

This is an educational project. Contributions are welcome for:

- **Feature improvements**: Better algorithms, new metrics
- **Visualization enhancements**: New plots, interactive elements
- **Documentation**: Tutorials, explanations, examples
- **Bug fixes**: Error handling, compatibility improvements

### Development Guidelines

1. **Educational focus**: Maintain clear learning objectives
2. **Code comments**: Explain DSP concepts thoroughly  
3. **Documentation**: Update README for new features
4. **Testing**: Verify with demo samples and real data
5. **Ethics**: Maintain medical disclaimers and educational purpose

## 📄 License

This project is for **educational use only**. 

- ✅ **Learning**: Study, modify, and learn from the code
- ✅ **Research**: Use for academic research projects
- ✅ **Teaching**: Incorporate into educational materials
- ❌ **Commercial**: No commercial or medical use
- ❌ **Diagnostic**: No medical diagnosis or treatment

## 🔗 Related Resources

### Academic Papers
- "Acoustic Analysis of Voice Quality" - Speech Communication
- "Cough Detection and Classification" - IEEE Transactions
- "Digital Health Biomarkers" - Nature Digital Medicine

### Online Courses
- Coursera: Digital Signal Processing
- edX: Biomedical Signal Processing  
- MIT OpenCourseWare: Signals and Systems

### Software Libraries
- [librosa](https://librosa.org/): Audio analysis library
- [Streamlit](https://streamlit.io/): Web app framework
- [matplotlib](https://matplotlib.org/): Scientific plotting
- [scipy](https://scipy.org/): Scientific computing

## 📞 Support

For questions about this educational project:

- **Technical issues**: Check the code comments and documentation
- **Educational questions**: Review the feature explanations in the app
- **Research applications**: Consult academic literature and datasets
- **Medical questions**: Always consult healthcare professionals

---

**Remember: This tool demonstrates audio signal processing capabilities, not medical diagnosis. Always consult healthcare professionals for medical concerns.**
"# voice-cough-mini-screener" 
