
# Thesis-Code-Book
Includes all the main python scripts used for the thesis project. 


# Chakra fMRI Analysis Code Base – README

This repository contains all scripts and accompanying documentation for the analysis of the fMRI case study on Ajna (Third-Eye) Chakra meditation. The goal is to assess brain activation differences between Ajna meditation and a Yogic Resting State (YRS) baseline, using ROI-based General Linear Model (GLM) analysis.

## Overview

The analysis was conducted using two pilot scans of a highly experienced yogi. Each scan included functional MRI data during Ajna meditation and YRS, preprocessed using fMRIprep (Pilot 1) and ME-ICA via TEDANA (Pilot 2). This code base supports:

- Construction of design matrices
- ROI mask generation and validation
- First-level GLM analysis per ROI (mean time series and voxelwise)
- Generation of statistical contrast maps and beta values

## Repository Structure



<img width="345" height="367" alt="image" src="https://github.com/user-attachments/assets/7d2475a3-f9ca-4caa-a917-41883242d52b" />






## Script Descriptions

### Design Matrix
The design matrix script code was written by Janick Bartels
- **build_design_matrix_P1.py / build_design_matrix_P2.py**  
  Create design matrices for GLM analysis, separately for each pilot. Includes task regressors (Ajna, YRS) and nuisance regressors (motion, WM, CSF, spikes). For Pilot 2, denoising was done via ICA, so no physiological regressors are included beyond spikes.

### Masks

- **[ROI]_MASK.py**  
  Define binary ROI masks from the Harvard-Oxford atlas using label matching and probabilistic thresholding (25%). Masks are initially in MNI space.

- **mni_Native_maskcomparison.py**  
  Visually and numerically compares ROI masks before and after transformation from MNI to native anatomical space. Compares the MNI visualisations to the Native visualisations.

- **native_maskoverlays.py**  
  Generates anatomical overlays to verify that native-space ROI masks align correctly with each subject’s T1-weighted image.

### ROI GLM Analysis

- **P1_ROI_GLM_ANALYSIS.py**  
  Runs ROI-level GLM analysis for Pilot 1. Computes both voxelwise contrast maps and ROI-mean time series statistics. 

- **P2_ROI_GLM_ANALYSIS_AvY.py**  
  Same as above, but for Pilot 2. Adjusted to work with multi-echo ICA-denoised data. 

### Utility Scripts

- **config.py**  
  Centralizes file paths and subject-specific information such as TR, image locations, and directory structures.

- **data.py**  
  Provides helper functions for image loading, ROI resampling, and data transformation. Supports other scripts modularly.


## Notes

- All analyses were run in subject-native space to preserve anatomical precision.
- ROI masks were transformed from MNI to native space using ANTs.
- Design matrices were created with Nilearn and matched to functional run lengths.  
- Pilot 1 and Pilot 2 are analyzed independently due to differences in scanning protocol and preprocessing pipeline.

