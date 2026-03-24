# Voice & Cough Mini-Screener

![Python](https://img.shields.io/badge/python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Coswara](https://img.shields.io/badge/data-Coswara-orange)

I built this to explore whether you can pull clinically meaningful 
signal out of a cough or voice recording using only acoustic features 
— no deep learning, no black box.

<img width="981" height="863" alt="image" src="https://github.com/user-attachments/assets/b111b533-b426-4b2c-915f-cefbd392f504" />

---

## What it does

Upload a WAV or MP3 file and it extracts the features that show up in 
respiratory research:

- **Jitter and Shimmer** — cycle-to-cycle instability in pitch and amplitude
- **Spectral Centroid** — where the energy sits in the frequency spectrum
- **Zero Crossing Rate** — separates voiced sound from turbulent airflow
- **Pitch tracking (F0)** — fundamental frequency over time

It outputs a waveform, spectrogram, pitch contour, and an irregularity 
score showing how far the sample deviates from stable phonation.

---

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload your own file or use the included Coswara samples.

---

## Dataset

Built on real recordings from the 
[Coswara dataset](https://coswara.iisc.ac.in/) — clinical audio 
collected at IISc Bangalore with healthy and COVID-positive subjects 
across cough, breathing, and vowel tasks. 67 processed samples are 
included. Full dataset at the link above.

---

## Stack

`librosa` · `streamlit` · `matplotlib`

---

## Status

Functional for feature extraction and visualization. Includes a simple 
rule-based audio-type suggestion (Cough / Voice / Breathing) as a 
first step toward a classification layer.

Not a diagnostic tool. Research and learning only.
