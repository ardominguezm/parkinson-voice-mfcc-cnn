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

