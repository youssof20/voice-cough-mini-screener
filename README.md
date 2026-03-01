# Voice & Cough Mini-Screener

Acoustic analysis tool that extracts clinical signal features from voice 
and cough recordings to identify irregularity patterns associated with 
respiratory conditions.

Built on real audio from the Coswara dataset — actual cough, breathing, 
and sustained vowel recordings collected during COVID-19 research at IISc Bangalore.

---

## What it actually does

Takes a WAV/MP3 file and extracts the features clinicians care about:

- **Jitter & Shimmer** — cycle-to-cycle pitch and amplitude instability
- **Spectral Centroid** — where the energy sits in the frequency spectrum
- **Zero Crossing Rate** — separates voiced sound from turbulent airflow
- **Pitch tracking (F0)** — fundamental frequency contour over time

Outputs a waveform, spectrogram, pitch contour, and an irregularity 
score that reflects how far the sample deviates from stable phonation.

---

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload any audio file or use the included Coswara samples to explore.

---

## Dataset

Uses real recordings from the 
[Coswara dataset](https://coswara.iisc.ac.in/) — 
a clinical audio collection from IISc Bangalore with healthy and 
COVID-positive subjects across cough, breathing, and vowel tasks.

67 processed samples included. Full dataset available at the link above.

---

## Stack

`librosa` · `streamlit` · `matplotlib`

---

## Status

Functional for feature extraction and visualization. 
Classification layer in progress.

*Not a diagnostic tool. For research and learning only.*
