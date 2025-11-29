# Automated Detection of Parkinson’s Disease from Voice Recordings  
### Using Convolutional Neural Networks and Synthetic MFCC Spectral Image Features

This repository contains the research code and dataset organization for the paper:

**Domínguez-Monterroza A., Mateos-Caballero A., Jiménez-Martín A.**  
*Automated Detection of Parkinson’s Disease from Voice Recordings Using Convolutional Neural Networks and Synthetic Spectral Image Features*,  Neural Computing and Applications, 2025, **In Review**.

## Acknowledgements.

This work was funded by grants PID2021-122209OB-C31 and RED2022-134540-T from MICIU/AEI/10.13039/501100011033

---

## Overview

This work proposes a complete pipeline for detecting Parkinson’s Disease (PD) using **Convolutional Neural Networks (CNNs)** trained on **MFCC spectral image features** extracted from raw audio signals of sustained phonation (/a/).  
Two experimental settings are implemented:

1. **Experiment 1 — Model trained and validated using real data**  
2. **Experiment 2 — Model trained using GAN-generated synthetic data and evaluated on real data**

Both experiments use:

- MFCC spectral representations  
- A deep CNN architecture  
- Accuracy, F1-score, Precision, Recall, and ROC-AUC  
- Multi-run cross-validation with mean, standard deviation, and coefficient of variation (CV)

---
###  Dataset source (PC-GITA)
The real voice dataset comes from:

- J. R. Orozco-Arroyave, J. D. Arias-Londoño, J. F. Vargas-Bonilla,  
  M. C. Gonzalez-Rátiva, and E. Nöth,  
  *New Spanish speech corpus database for the analysis of people suffering from Parkinson’s disease*,  
  Proc. 9th Int. Conf. Language Resources and Evaluation, 2014.

The dataset includes:

- **100 Colombian speakers**  
  - 50 PD  
  - 50 healthy controls  
- Each participant phonated */a/* three times → **300 total recordings (150 PD, 150 HC)**.

**Synthetic data generation Dataset source**:
- M. Rey-Paredes, C. J. Pérez, A. Mateos-Caballero,  
  *Time Series Classification of Raw Voice Waveforms for Parkinson’s Disease Detection Using Generative Adversarial Network-Driven Data Augmentation*, IEEE Open Journal of the Computer Society, 2025.
---

## Repository Structure

```text
parkinson-voice-mfcc-cnn/
│
├── raw/
│   ├── control/                 # Raw voice CSV files (healthy)
│   └── parkinson/               # Raw voice CSV files (PD)
│
├── processed/
│   ├── mfcc_images.npy          # MFCC images used for CNN input
│   ├── labels.npy               # Labels (0=control, 1=PD)
│   └── README.md
│
├── metadata/
│   ├── dataset_description.md   # PD dataset documentation
│   ├── participants_info.csv    # Speaker metadata
│   └── LICENSE
│
├── code/
│   ├── 01_extract_mfcc.py
│   ├── 02_build_cnn.py
│   ├── 03_cross_validation.py
│   ├── 04_visualization.py
│   └── full_pipeline.py         # Complete experiment script
│
├── analysis/
│   ├── metrics_real_data.csv
│   ├── metrics_synthetic_eval.csv
│   ├── mfcc_examples/
│   └── results_summary.md
│
├── CITATION.cff
└── README.md
