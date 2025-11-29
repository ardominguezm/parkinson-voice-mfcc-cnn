# Automated Detection of Parkinson’s Disease from Voice Recordings Using MFCC Spectral Features and Convolutional Neural Networks
 
This repository accompanies the manuscript:

Domínguez-Monterroza A., Mateos-Caballero A., Jiménez-Martín A.
*Automated Detection of Parkinson’s Disease from Voice Recordings Using Convolutional Neural Networks and Synthetic Spectral Image Features*. Neural Computing and Applications, 2025. In review.


It includes raw data organization, MFCC extraction pipeline, CNN training, cross-validation experiments, and result analysis.

The project follows:
- **Structured data folders**: raw → processed → analysis
- **Version-controlled code**
- **Metadata and documentation for reproducibility**
- **Automatic archival and DOI via Zenodo**

---

##  Project Description

The goal of this study is to classify Parkinson’s Disease (PD) vs Healthy Control (HC) subjects using MFCC-based spectral images derived from sustained phonation of the vowel */a/*. A deep Convolutional Neural Network (CNN) was trained on MFCC images extracted from real Colombian patients.

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

## 📁 Repository Structure
parkinson-voice-mfcc-cnn/
│
├── raw/
│   ├── control/                      # Raw CSV signals of healthy controls
│   ├── parkinson/                    # Raw CSV signals of PD patients
│
├── processed/
│   ├── mfcc_images.npy              # MFCC matrices after padding
│   ├── labels.npy                   # Corresponding labels (0=HC, 1=PD)
│
├── metadata/
│   ├── dataset_description.md       # Source, acquisition, annotation details
│   ├── participants_info.csv        # Basic demographics if available
│   ├── LICENSE                      # License information
│
├── code/
│   ├── 01_extract_mfcc.py           # Signal processing and MFCC extraction
│   ├── 02_build_cnn.py              # CNN architecture definition
│   ├── 03_cross_validation.py       # Stratified 10-fold CV script
│   ├── 04_visualization.py          # MFCC image visualization
│   ├── full_pipeline.py             # Unified end-to-end reproducible workflow
│
├── analysis/
│   ├── metrics_real_data.csv        # Accuracy, F1, precision, recall, ROC-AUC
│   ├── mfcc_examples/               # Figures for healthy and PD subjects
│   ├── results_summary.md           # Statistical summary and discussion
│
├── README.md                        # Main documentation (this file)
└── CITATION.cff                     # For Zenodo DOI attribution

