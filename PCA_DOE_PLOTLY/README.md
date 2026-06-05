# PCA + GP Surrogate · DoE Compressor Tool  (web edition)

Interactive Plotly Dash replacement for the original matplotlib GUI.

## Files

| File | Purpose |
|------|---------|
| `app3.py` | Dash web app — all layout, callbacks, and plot logic |
| `pca_core.py` | Shared PCA / GP / data-loading (unchanged logic from original) |

## Setup

```bash
pip install dash plotly scikit-learn numpy pandas librosa tqdm
```

## Run

```bash
# Place in the same directory as your data:
#   doe_params.csv
#   stl_files/DOE_*.stl
#   data/doe_data/*.csv
#   audio/doe_data/*.wav   (optional — stored for future audio tab)

python app.py
# → opens http://localhost:8050 automatically
```

## Tabs

| Tab | Description |
|-----|-------------|
| Tab 1 — ψ(φ) | Pressure-rise coefficient PCA surrogate |
| Tab 2 — η(φ) | Efficiency PCA surrogate |
| Tab 3 — 🎵 Audio | **Coming soon** — librosa data already loaded at startup |

## Features

- **8 parameter sliders** with live drag updates
- **Parameter sweep** — select any parameter to overlay a coloured family of curves
- **PCA mode count** radio (k = 1–6) — refits GP surrogates on the fly
- **GP uncertainty bands** ±1σ and ±2σ propagated through mode scores
- **Scree plot** with cumulative variance and 95% threshold
- **Principal mode shapes** plot
- **Model info panel** — GP scores, LOO RMS, variance captured
- **Export CSV** — downloads φ, predicted curve, mean curve, and all mode shapes

## Audio tab (future)

`pca_core.py` loads each run's `.wav` file via `librosa` at startup and stores
`audio_data` (np.ndarray) and `sample_rate` (int) in every run dict.
The audio tab placeholder lists planned features:
spectrogram viewer, MFCC comparison, tonal/broadband decomposition, 1/3-octave SPL.
