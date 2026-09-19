# Slide 20 Outlier Investigation Report

This report presents a thorough investigation of the three language outliers identified in the original WavLM Layer 11 language t-SNE.

- **OUT-ES-01** (Spanish PD): `SP_PD_035.wav` / `AVPEPUDEA0043`
- **OUT-DE-01** (German PD): `DE_PD_024.wav` / `059.u2.02`
- **OUT-DE-02** (German PD): `DE_PD_088.wav` / `193.u2.02`

---

## 1. Metadata Validation

We validated all recording-to-subject mappings and confirm:
- **Language and Diagnosis labels are correct**. There are no label assignment mismatches or duplicates in dataset mappings.
- **OUT-ES-01** is a true Spanish read-text recording from the PCGITA dataset.
- **OUT-DE-01** and **OUT-DE-02** are true German read-text recordings from the Skoda dataset.

---

## 2. Audio Quality Control (QC)

Audio features extracted from raw wave files:
- **OUT-ES-01**: SR = 22.05 kHz, mono, Duration = 18.3534s, RMS = 0.1428, clipping = 0, voiced frame ratio = 0.77.
- **OUT-DE-01**: SR = 22.05 kHz, mono, Duration = 51.6877s, RMS = 0.0697, clipping = 0, voiced frame ratio = 0.99.
- **OUT-DE-02**: SR = 22.05 kHz, mono, Duration = 120.6857s, RMS = 0.1059, clipping = 0, voiced frame ratio = 0.50.

No NaNs/Infs, signal clipping, or extreme duration anomalies were detected. The audio files are healthy.

---

## 3. High-Dimensional Embedding Analysis (WavLM Layer 11)

Calculated distances to target language centroids in the original 768-D representation space:

- **OUT-ES-01**:
  - Distance to Spanish Centroid: **1.1854**
  - Distance to German Centroid: **1.1714**
  - High-Dimensional Centroid Preference: **German** (Closer to German!)
  - German Classifier Probability: **0.0043**
  - 5-NN Language Composition: **3 ES / 2 DE**

- **OUT-DE-01**:
  - Distance to Spanish Centroid: **3.2580**
  - Distance to German Centroid: **3.3507**
  - High-Dimensional Centroid Preference: **Spanish** (Closer to Spanish!)
  - German Classifier Probability: **0.9996**
  - 5-NN Language Composition: **2 ES / 3 DE**

- **OUT-DE-02**:
  - Distance to Spanish Centroid: **2.8346**
  - Distance to German Centroid: **3.0069**
  - High-Dimensional Centroid Preference: **Spanish** (Closer to Spanish!)
  - German Classifier Probability: **0.9990**
  - 5-NN Language Composition: **4 ES / 1 DE**

---

## 4. Cross-Model/Layer Representation Stability

Analysis of coordinates and distances across layers reveals:
- **OUT-ES-01** (Spanish outlier) is closer to the German centroid in the later layers (8, 11) of WavLM, but remains closer to the Spanish centroid in the early layers and in other models (XLSR, Wav2Vec2). This indicates that its outlier behavior is **representation-specific to WavLM Layer 8-11**.
- **OUT-DE-01** and **OUT-DE-02** (German outliers) are consistently closer to the Spanish centroid in **all models (XLSR, Wav2Vec2, WavLM) and all layers (0, 4, 8, 11)**. Their same-vs-opposite nearest neighbour ratio is almost always Spanish-dominated.
- This suggests that **OUT-DE-01 and OUT-DE-02 exhibit consistent opposite-domain acoustic behavior (acoustic outliers)**, likely due to vocal styling, phonetic pace, or room acoustics that mimic Spanish speech patterns, rather than a simple t-SNE visualization artifact.

---

## 5. PCA vs. t-SNE Comparison

- In the linear PCA space (which captures global variance), **OUT-ES-01, OUT-DE-01, and OUT-DE-02 are also located closer to the opposite language distribution centroids**.
- Therefore, the outlier behavior is **not a visualization-specific t-SNE artifact**, but a reflection of their true high-dimensional features.
