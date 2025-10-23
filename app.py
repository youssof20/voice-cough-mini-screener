"""
Voice & Cough Mini-Screener
============================

A Streamlit web application that analyzes voice and cough audio files using
digital signal processing techniques to extract acoustic biomarkers.

IMPORTANT DISCLAIMER:
This application is for EDUCATIONAL and DEMONSTRATION purposes only.
It is NOT a medical device and should NOT be used for diagnostic purposes.
The "risk score" is purely demonstrative and has no clinical validity.

Author: Voice & Cough Mini-Screener Project
License: Educational use only - Non-diagnostic
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
import os
import tempfile

# Import our custom modules
from utils.audio_features import extract_all_features, calculate_risk_score, load_audio
from utils.visualizations import (
    create_waveform_plot, 
    create_spectrogram_plot, 
    create_pitch_curve_plot,
    create_combined_plot,
    fig_to_base64
)


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Voice & Cough Mini-Screener",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .risk-indicator {
        background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%);
        height: 30px;
        border-radius: 15px;
        position: relative;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Voice & Cough Mini-Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Digital Signal Processing & Acoustic Analysis Demo</p>', unsafe_allow_html=True)
    
    # Important disclaimer
    st.markdown("""
    <div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This application is for EDUCATIONAL and DEMONSTRATION purposes only.</strong></p>
    <ul>
    <li>It is <strong>NOT a medical device</strong> and should <strong>NOT be used for diagnostic purposes</strong></li>
    <li>The "risk score" is purely demonstrative and has <strong>no clinical validity</strong></li>
    <li>Always consult healthcare professionals for medical concerns</li>
    <li>This tool demonstrates audio signal processing capabilities, not medical diagnosis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Audio Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Upload a voice or cough audio file for analysis"
        )
        
        # Real data samples section
        st.header("🎵 Real Audio Samples")
        st.write("Try these real audio samples from the Coswara dataset:")
        
        # Get real data samples
        real_samples = {}
        if os.path.exists("data/processed"):
            processed_files = [f for f in os.listdir("data/processed") if f.endswith('.wav')]
            # Show first 10 real samples with descriptive names
            for i, file in enumerate(processed_files[:10]):
                # Extract type from filename
                if 'cough' in file:
                    sample_type = "Cough"
                elif 'breathing' in file:
                    sample_type = "Breathing"
                else:
                    sample_type = "Voice"
                real_samples[f"{sample_type} Sample {i+1}"] = f"data/processed/{file}"
        
        if not real_samples:
            st.warning("No real audio samples found. Please run the data processing script first.")
            st.code("python utils/data_integration.py --process-coswara")
        
        selected_demo = st.selectbox("Select Sample:", ["None"] + list(real_samples.keys()))
        
        if selected_demo != "None":
            sample_path = real_samples[selected_demo]
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    st.download_button(
                        label=f"Download {selected_demo}",
                        data=f.read(),
                        file_name=f"{selected_demo.lower().replace(' ', '_')}.wav",
                        mime="audio/wav"
                    )
        
        # Analysis parameters
        st.header("⚙️ Analysis Settings")
        st.write("**Sampling Rate:** 22,050 Hz (standard for speech)")
        st.write("**Analysis Window:** Full audio duration")
        st.write("**Features:** Pitch, Energy, Spectral, Stability")
        
        # Links to datasets
        st.header("📊 Open Datasets")
        st.write("For real testing, try these open datasets:")
        st.markdown("- [COUGHVID](https://coughvid.epfl.ch/)")
        st.markdown("- [Coswara](https://coswara.iisc.ac.in/)")
        st.markdown("- [COVID-19 Sounds](https://covid-19-sounds.org/)")
    
    # Main content area
    if uploaded_file is not None or selected_demo != "None":
        
        # Determine which file to analyze
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name
            file_name = uploaded_file.name
        else:
            # Use selected sample
            file_path = real_samples[selected_demo]
            file_name = f"{selected_demo}.wav"
        
        try:
            # Extract features
            with st.spinner("Analyzing audio features..."):
                features_df, feature_descriptions = extract_all_features(file_path)
                risk_score = calculate_risk_score(features_df)
            
            # Load audio for visualization
            audio, sr = load_audio(file_path)
            
            # Display results
            st.success(f"✅ Successfully analyzed: **{file_name}**")
            
            # Create two-column layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.header("📊 Extracted Features")
                
                # Display features in a nice table
                display_df = features_df.copy()
                display_df.columns = [feature_descriptions.get(col, col) for col in display_df.columns]
                
                st.dataframe(
                    display_df.T,
                    use_container_width=True,
                    height=400
                )
                
                # Acoustic Analysis Summary
                st.header("🎯 Acoustic Analysis Summary")
                st.markdown("*This is a demonstration metric only - NOT diagnostic*")
                
                # Create risk gauge using plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Analysis Score"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Analysis interpretation
                if risk_score < 30:
                    analysis_level = "Low Variation"
                    analysis_color = "green"
                elif risk_score < 70:
                    analysis_level = "Moderate Variation" 
                    analysis_color = "orange"
                else:
                    analysis_level = "High Variation"
                    analysis_color = "red"
                
                st.markdown(f"**Analysis Level:** <span style='color: {analysis_color}'>{analysis_level}</span>", unsafe_allow_html=True)
                st.markdown(f"**Score:** {risk_score:.1f}/100")
            
            with col2:
                st.header("📈 Audio Visualizations")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Combined View", "Waveform", "Spectrogram"])
                
                with tab1:
                    # Combined plot
                    fig_combined = create_combined_plot(audio, sr, f"Analysis: {file_name}")
                    st.pyplot(fig_combined)
                
                with tab2:
                    # Waveform only
                    fig_wave = create_waveform_plot(audio, sr, f"Waveform: {file_name}")
                    st.pyplot(fig_wave)
                
                with tab3:
                    # Spectrogram only
                    fig_spec = create_spectrogram_plot(audio, sr, f"Spectrogram: {file_name}")
                    st.pyplot(fig_spec)
            
            # Feature explanations
            st.header("🔬 Feature Explanations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🎵 Pitch Features")
                st.markdown("""
                - **Mean Pitch (F0):** Fundamental frequency of voice
                - **Pitch Std:** Variation in pitch over time
                - **Jitter:** Micro-variations in pitch (voice stability)
                """)
            
            with col2:
                st.subheader("🔊 Energy Features")
                st.markdown("""
                - **RMS Energy:** Average loudness/amplitude
                - **Peak Amplitude:** Maximum volume level
                - **Shimmer:** Micro-variations in amplitude
                """)
            
            with col3:
                st.subheader("📊 Spectral Features")
                st.markdown("""
                - **Spectral Centroid:** "Brightness" of sound
                - **Zero Crossing Rate:** Voice vs noise indicator
                - **Spectral Bandwidth:** Frequency spread
                """)
            
            # Technical details
            with st.expander("🔧 Technical Details"):
                st.markdown(f"""
                **Audio Properties:**
                - Duration: {len(audio)/sr:.2f} seconds
                - Sample Rate: {sr} Hz
                - Samples: {len(audio):,}
                - Channels: Mono
                
                **Analysis Method:**
                - Pitch tracking using librosa's piptrack algorithm
                - Spectral analysis with 2048-point FFT
                - Frame-based processing with 512-sample hop length
                - Standard speech analysis parameters (50-400 Hz pitch range)
                """)
            
            # Clean up temporary file
            if uploaded_file is not None and os.path.exists(file_path):
                os.unlink(file_path)
                
        except Exception as e:
            st.error(f"❌ Error analyzing audio: {str(e)}")
            st.markdown("Please try a different audio file or use one of the demo samples.")
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        ## 🎯 Welcome to the Voice & Cough Mini-Screener!
        
        This application demonstrates **digital signal processing** techniques used in:
        - **Speech Analysis** - Voice quality assessment using DSP
        - **Respiratory Analysis** - Cough pattern analysis with spectral methods
        - **Acoustic Biomarkers** - Voice stability indicators from signal processing
        - **Audio Analysis** - Real-time audio feature extraction
        
        ### 🚀 Getting Started:
        1. **Upload an audio file** using the sidebar (WAV, MP3, M4A supported)
        2. **Try real samples** - actual audio from the Coswara medical dataset
        3. **View results** - acoustic features, visualizations, and analysis score
        4. **Learn more** - explore feature explanations and technical details
        
        ### 📚 Educational Value:
        This tool teaches you about:
        - Digital signal processing fundamentals
        - Audio feature extraction techniques
        - Biomedical signal analysis
        - Web application development with Streamlit
        
        ### ⚠️ Remember:
        This is a **demonstration tool only** - not for medical diagnosis!
        """)
        
        # Show sample feature table
        st.subheader("📋 Sample Feature Output")
        sample_features = pd.DataFrame({
            'Feature': ['Duration (s)', 'Mean Pitch (Hz)', 'RMS Energy', 'Jitter (%)', 'Shimmer (%)'],
            'Typical Range': ['1-10', '80-300', '0.01-0.5', '0.1-2.0', '0.5-5.0'],
            'Clinical Relevance': [
                'Audio length',
                'Voice pitch (F0)',
                'Loudness level', 
                'Voice stability',
                'Amplitude variation'
            ]
        })
        st.dataframe(sample_features, use_container_width=True)


if __name__ == "__main__":
    main()
