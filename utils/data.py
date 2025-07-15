# utils/data.py
# This is the data file for VajraCode. 

# It provides functions for loading, cleaning, and saving fMRI data from PILOT 1 (P1) and PILOT 2 (P2). 
# It uses the filepaths and parameters defined in `utils.config`. 

# USAGE
# 1) Run the script so that it saves the baseline, meditation and full timeseries at the roi level (7 and 17 yeo networks) 
# as a pickle file, with smoothing, FD scrubbing and confound regression applied as defined in the save_series function.
# 2) Load the resulting time series data for analysis, calling the `load_series` function in other scripts.

# KEY FEATURES
# - Applies confound regression to the data (default: 6 motion parameters + top 5 CompCor).
# - Supports both the **7 Yeo** and **17 Yeo** network versions of the **Schaefer 2018 atlas (default: 7 Yeo)**.
# - Includes a **smoothing parameter (default: 6mm FWHM)**.
# - Allows extraction at both **ROI and voxel levels**.
# - Performs **framewise displacement (FD) scrubbing** using cubic spline interpolation to replace bad volumes.
# - Saves all `.pkl` time series files into a dedicated subfolder: **output/pkl_timeseries/**.

# REQUIREMENTS
# - fMRI data must be preprocessed with fMRIprep (and tedana for ME data).
# - The folder structure must mirror the CHAKRA_PREP structure found in the SurfDrive folder.

# SECTIONS:
# 1) Imports and Global Variables
# 2) Helper Functions
# 3) Main Functions
#    - get_series: Load, slice, clean, and optionally smooth time series data
#    - save_series: Extracts and saves time series to pickle files in output/pkl_timeseries/
#    - load_series: Loads previously saved pickle files
# 4) Main Execution (batch processing)


##############################################
###  1) IMPORTS AND GLOBAL VARIABLES       ###
##############################################

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nibabel as nib
import pandas as pd
import numpy as np
import pickle
import re
from scipy.interpolate import CubicSpline
from nilearn.image import index_img, clean_img, load_img, mean_img, math_img, resample_to_img
from nilearn import datasets, maskers
from nilearn.masking import apply_mask, unmask
from nilearn.signal import clean


# Import configuration variables
from utils.config import (
    FD_THRESHOLD,
    OUTPUT_PATH,
    P1_PATH, P1_FUNC_IMG_PATHS_NATIVE, P1_FUNC_IMG_PATHS_MNI, P1_BRAIN_MASK_PATHS_NATIVE_REFINED, P1_TR, P1_CONFOUNDS, P1_TIMINGS, P1_SCHAEFER_NAT_7YEO, P1_SCHAEFER_NAT_17YEO, P1_GM_MASK_NATIVE,
    P2_PATH, P2_FUNC_IMG_PATHS_NATIVE, P2_FUNC_IMG_PATHS_MNI, P2_BRAIN_MASK_PATHS_NATIVE, P2_TIMINGS, P2_CONFOUNDS, P2_MEICA_CONFOUNDS, P2_TR, P2_SCHAEFER_NAT_7YEO , P2_SCHAEFER_NAT_17YEO, P2_GM_MASK_NATIVE, P2_MEICA_FUNC_IMG_PATHS_NATIVE,
)


#########################################
# 1.1) Load Schaefer Atlas (MNI Space) ##
#########################################
def get_schaefer_atlas(yeo_networks=7):
    return datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=yeo_networks, resolution_mm=2) 

# Preload MNI Schaefer atlas images
atlas_img_7yeo = get_schaefer_atlas(7).maps
atlas_img_17yeo = get_schaefer_atlas(17).maps

###########################################
# 1.2) Load Schaefer Atlas (Native Space) #
###########################################
def get_schaefer_atlas_native(pilot="P1", yeo_networks=7):
    """
    Load the Schaefer atlas that has been resampled to the participant’s native space.

    Parameters:
    - pilot (str): Either "P1" (Pilot 1) or "P2" (Pilot 2).
    - yeo_networks (int): Choose between 7 (default) or 17 networks.

    Returns:
    - nibabel Nifti1Image: Native space Schaefer atlas image.
    """
    if pilot == "P1":
        atlas_path = P1_SCHAEFER_NAT_7YEO if yeo_networks == 7 else P1_SCHAEFER_NAT_17YEO
    elif pilot == "P2":
        atlas_path = P2_SCHAEFER_NAT_7YEO if yeo_networks == 7 else P2_SCHAEFER_NAT_17YEO
    else:
        raise ValueError("Invalid pilot selection. Choose 'P1' or 'P2'.")

    return nib.load(atlas_path)

# Preload native Schaefer atlases for each pilot
atlas_img_7yeo_native_P1 = get_schaefer_atlas_native("P1", 7)
atlas_img_17yeo_native_P1 = get_schaefer_atlas_native("P1", 17)
atlas_img_7yeo_native_P2 = get_schaefer_atlas_native("P2", 7)
atlas_img_17yeo_native_P2 = get_schaefer_atlas_native("P2", 17)

#########################################
## 1.3) Get Masker in Native/MNI Space ##
#########################################
def get_masker(pilot="P1", space="native", yeo_networks=7, smoothing_fwhm=6, mask_img=None):
    """
    Get a NiftiLabelsMasker for extracting time series from fMRI data.
    
    Parameters:
    - pilot (str): "P1" or "P2" (default: "P1") to select the correct native Schaefer atlas.
    - space (str): "MNI" or "native" (default: "native").
    - yeo_networks (int): Choose between 7 or 17 networks.
    - smoothing_fwhm (int): Smoothing kernel size in mm (default: 6mm).
    
    Returns:
    - NiftiLabelsMasker object
    """

    if space == "MNI":
        atlas_img = atlas_img_7yeo if yeo_networks == 7 else atlas_img_17yeo
    elif space == "native":
        if pilot == "P1":
            atlas_img = atlas_img_7yeo_native_P1 if yeo_networks == 7 else atlas_img_17yeo_native_P1
        elif pilot == "P2":
            atlas_img = atlas_img_7yeo_native_P2 if yeo_networks == 7 else atlas_img_17yeo_native_P2
        else:
            raise ValueError("Invalid pilot selection. Choose 'P1' or 'P2'.")
    else:
        raise ValueError("Invalid space argument. Choose 'MNI' or 'native'.")

    return maskers.NiftiLabelsMasker(
        labels_img=atlas_img,
        t_r=2.2,
        standardize=True,
        detrend=True,      
        high_pass=0.006,    # preserve slow meditation signal
        low_pass=0.09,
        smoothing_fwhm=smoothing_fwhm,
        memory='nilearn_cache',
        verbose=5,
        mask_img=mask_img 
    )

##############################################
###         2) HELPER FUNCTIONS            ###
##############################################

def short_confounds_name(confound_columns):
    """
    Create a short descriptor string for the selected confounds.
    - Motion parameters (6): 'mot6'
    - White matter: 'wm'
    - Global signal: 'gs'
    - First 5 anatomical CompCor components: 'Comp5'
    - Any other confounds are added as-is.

    Example:
        ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','white_matter','csf']
        -> 'mot6-wm-csf'
    """

    motion_set = {'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'}
    columns_set = set(confound_columns)

    confound_labels = []

    # Check if all 6 motion parameters are present
    if motion_set.issubset(columns_set):
        confound_labels.append('mot6')
        columns_set = columns_set - motion_set

    # Detect and count a_comp_cor components
    compcor_components = {col for col in columns_set if re.match(r'a_comp_cor_\d+', col)}
    if compcor_components:
        confound_labels.append(f'Comp{len(compcor_components)}')
        columns_set -= compcor_components

    # Check for white_matter and csf
    if 'white_matter' in columns_set:
        confound_labels.append('wm')
        columns_set.remove('white_matter')
    if 'global_signal' in columns_set:
        confound_labels.append('gs')
        columns_set.remove('global_signal')

    # Add any remaining confounds as-is
    # (sort for consistent ordering)
    for extra in sorted(columns_set):
        confound_labels.append(extra)

    if len(confound_labels) == 0:
        return 'none'
    else:
        return '-'.join(confound_labels)

def expand_indices(indices, max_len):
    """
    Expands a list of timepoint indices by ±1 (previous and next) for scrubbing,
    clipped to the valid range [0, max_len).
    
    Parameters
    ----------
    indices : array-like
        List or array of time indices to expand.
    max_len : int
        Maximum allowed index (usually the number of timepoints).
    
    Returns
    -------
    List[int]
        Sorted list of unique, valid indices including neighbors.
    """
    expanded = set()
    for i in indices:
        for j in [i - 1, i, i + 1]:
            if 0 <= j < max_len:
                expanded.add(j)
    return sorted(expanded)

def nan_scrub_timepoints(time_series, bad_indices):
    """
    Replaces specified timepoints in the time series with NaNs.
    
    Parameters
    ----------
    time_series : np.ndarray
        (Time x Features) array of fMRI signal.
    bad_indices : array-like
        Indices of time points to scrub.
    
    Returns
    -------
    scrubbed_time_series : np.ndarray
        Time series with NaNs inserted at scrubbed volumes.
    """
    scrubbed_series = time_series.copy()
    scrubbed_series[bad_indices, :] = np.nan
    return scrubbed_series

def interpolate_bad_volumes(time_series, bad_indices):
    """
    Performs cubic spline interpolation to replace high-FD volumes.
    
    Parameters
    ----------
    time_series : np.ndarray
        (Time x Features) array of fMRI signal.
    bad_indices : np.ndarray
        Indices of time points with high FD.
    
    Returns
    -------
    interpolated_time_series : np.ndarray
        Time series with interpolated values at high-FD volumes.
    """
    n_timepoints, n_features = time_series.shape
    good_indices = np.setdiff1d(np.arange(n_timepoints), bad_indices)

    # Interpolate for each feature (ROI or voxel)
    interpolated_series = time_series.copy()
    for i in range(n_features):
        spline = CubicSpline(good_indices, time_series[good_indices, i])
        interpolated_series[bad_indices, i] = spline(bad_indices)
    
    return interpolated_series

def report_nan_stats(time_series, label=""):
    """
    Prints summary stats on NaNs in a time series array.

    Parameters
    ----------
    time_series : np.ndarray
        Time x ROI or Time x Voxel data
    label : str, optional
        Label to describe the scan or condition (e.g., 'baseline', 'meditation')
    """
    n_timepoints = time_series.shape[0]
    n_nan_vols = np.any(np.isnan(time_series), axis=1).sum()
    percent_nan = 100 * n_nan_vols / n_timepoints
    print(f"[Scrubbing Report] {label}: {n_timepoints} volumes total | {n_nan_vols} NaN (scrubbed) volumes | {percent_nan:.2f}% NaNs")


def seconds_to_TR_indices(start_sec, end_sec, TR):
    """
    Convert onset and offset times (in seconds) into fMRI volume indices (TRs) for General Linear Model (GLM) analysis.
    This function ceils and floors the timings of the scan conditions into TR volume numbers. 
    This prepares the timings for usage in GLMs and makes sure only TRs that are fully part of the silent part of the condition are included.
    """
    onset_TR = int(np.ceil(start_sec / TR))
    offset_TR = int(np.floor(end_sec / TR))
    return onset_TR, offset_TR


##############################################
###          3) MAIN FUNCTIONS             ###
##############################################

def get_series(
    scan,
    fd_scrubbing=True,
    level='roi', 
    yeo_networks=7, 
    smoothing_fwhm=6,
    space="native",
    confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],    
    mask=None,  # Options: None, "GM", "BM", "GMBM"
    mask_threshold=0.5,
):
    """
    Load and return baseline and meditation fMRI time series for a given scan.

    This function performs:
        1) Pilot and path selection based on scan ID.
        2) Loading of functional and confound data.
        3) Mask application (optional):
            - mask=None: No masking applied (full volume or default atlas)
            - mask="GM": Gray matter mask only, thresholded
            - mask="BM": scan-specific brain mask
            - mask="GMBM": Gray matter AND scan-specific brain mask (default legacy)
        4) Extraction at ROI or voxel level using appropriate masker
        5) Confound regression (optional)
        6) Framewise displacement scrubbing:
            - If True: Replace bad volumes (±1) with NaN
            - If 'interpol': Replace bad volumes (±1) with spline interpolation
            - If False: No scrubbing
        7) Returns baseline, meditation, and full time series.

    Parameters
    ----------
    scan : str
        Identifier for the specific scan (e.g., 'P1S2', 'P2S7MEICA').

    fd_scrubbing : {True, False, 'interpol'}, optional
    If 'interpol', volumes with FD > FD_THRESHOLD are interpolated using spline. (e.g. for visualization purposes)
    If True, volumes with FD > FD_THRESHOLD (±1) are replaced with NaNs.        (for data analysis)
    If False, no scrubbing is applied.

    level : {'roi', 'voxel'}, optional
        Granularity of the data. 'roi' (default) returns atlas-based ROI time series,
        'voxel' returns voxel-level time series.

    yeo_networks : int
            Number of Yeo networks (7 or 17).

        smoothing_fwhm : float
            FWHM for spatial smoothing.

        space : {'native', 'mni'}
            Image space.

        confound_columns : list of str, optional
            Confound columns to be regressed out. Default includes motion parameters,
            white matter, and CSF.

        mask : str or None
                Masking strategy: None, 'GM', 'GS', 'GMGS', 'GMBM', 'GMGSBM'.
        mask_threshold : float
                Threshold for gray matter probability mask.


    Returns
    -------
    baseline_time_series : np.ndarray
        Cleaned baseline time series (time x ROI or voxel).
    meditation_time_series : np.ndarray
        Cleaned meditation time series (time x ROI or voxel).
    full_time_series : np.ndarray
        Cleaned full time series (time x ROI or voxel).


    """
    if confound_columns is None:
        confound_columns = []

    print(f"[GET] Scan {scan}")

    # Decide whether we're dealing with P1 or P2 scans and select correct FUNC_IMG_PATHS
    if 'P2' in scan:
        pilot = "P2"
        PATH = P2_PATH
        TIMINGS = P2_TIMINGS
        TR = P2_TR
    # Determine whether it's MEICA or standard multi-echo (or fallback to default)
        if scan.endswith("MEICA"):
            FUNC_IMG_PATHS = P2_MEICA_FUNC_IMG_PATHS_NATIVE if space == "native" else P2_MEICA_FUNC_IMG_PATHS_MNI
            BRAIN_MASK_PATHS = P2_BRAIN_MASK_PATHS_NATIVE if space == "native" else P2_BRAIN_MASK_PATHS_MNI
            GM_MASK = P2_GM_MASK_NATIVE if space == "native" else P2_GM_MASK_MNI
            CONFOUNDS = P2_MEICA_CONFOUNDS
        else:
            FUNC_IMG_PATHS = P2_FUNC_IMG_PATHS_NATIVE if space == "native" else P2_FUNC_IMG_PATHS_MNI
            BRAIN_MASK_PATHS = P2_BRAIN_MASK_PATHS_NATIVE if space == "native" else P2_BRAIN_MASK_PATHS_MNI
            GM_MASK = P2_GM_MASK_NATIVE if space == "native" else P2_GM_MASK_MNI
            CONFOUNDS = P2_CONFOUNDS

        
    elif 'P1' in scan:
        pilot = "P1"
        PATH = P1_PATH
        FUNC_IMG_PATHS = P1_FUNC_IMG_PATHS_NATIVE if space == "native" else P1_FUNC_IMG_PATHS_MNI
        BRAIN_MASK_PATHS = P1_BRAIN_MASK_PATHS_NATIVE_REFINED if space == "native" else P1_BRAIN_MASK_PATHS_MNI   # MNI mask is as is, not refined
        GM_MASK = P1_GM_MASK_NATIVE if space == "native" else P1_GM_MASK_MNI
        TIMINGS = P1_TIMINGS
        CONFOUNDS = P1_CONFOUNDS
        TR = P1_TR
    else:
        raise ValueError(f"Scan '{scan}' not recognized as P1 or P2.")


    # Load functional image
    func_img_path = os.path.join(PATH, FUNC_IMG_PATHS[scan])
    confounds_path = os.path.join(PATH, CONFOUNDS[scan])
    func_img = nib.load(func_img_path)
    confounds_df = pd.read_csv(confounds_path, sep='\t')

    # Log shape of functional image 
    n_voxels_total = np.prod(func_img.shape[:3])
    n_timepoints = func_img.shape[3]
    print(f"[SHAPE] {scan} Functional image shape: {func_img.shape} (voxels: {n_voxels_total}, timepoints: {n_timepoints})")



    # Extract framewise displacement (FD) and identify high-FD volumes
    fd = confounds_df['framewise_displacement'].fillna(0).values
    bad_indices = np.where(fd > FD_THRESHOLD)[0]
  #  print(f"[DEBUG] bad_indices: {len(bad_indices)} timepoints to scrub")

    # Identify baseline condition indices
    baseline_start_time_sec = TIMINGS[scan]['baseline']['start']
    baseline_end_time_sec = TIMINGS[scan]['baseline']['end']
    baseline_start_index = int(baseline_start_time_sec / TR)
    baseline_end_index = int(baseline_end_time_sec / TR)
    baseline_indices = np.arange(baseline_start_index, baseline_end_index)

    # Identify meditation condition indices
    meditation_start_time_sec = TIMINGS[scan]['meditation']['start']
    meditation_end_time_sec = TIMINGS[scan]['meditation']['end']
    meditation_start_index = int(meditation_start_time_sec / TR)
    meditation_end_index = int(meditation_end_time_sec / TR)
    meditation_indices = np.arange(meditation_start_index, meditation_end_index)

    # For the full time series, simply take all volumes
    full_indices = np.arange(func_img.shape[-1])



    # Prepare confounds if requested
    if len(confound_columns) > 0:
        baseline_confounds = confounds_df.iloc[baseline_indices][confound_columns]
        meditation_confounds = confounds_df.iloc[meditation_indices][confound_columns]
        full_confounds = confounds_df.iloc[full_indices][confound_columns]

    else:
        baseline_confounds = None
        meditation_confounds = None
        full_confounds = None


    # Extract time series data at the chosen level
    if level == 'roi':
        # ROI-level time series
        baseline_clean_func_img = index_img(func_img, baseline_indices)
        meditation_clean_func_img = index_img(func_img, meditation_indices)
        full_clean_func_img = index_img(func_img, full_indices)

        # Load brain and GM masks
        gm_mask = load_img(GM_MASK)
        brain_img = load_img(os.path.join(PATH, BRAIN_MASK_PATHS[scan]))

        # Compose mask logic
        if mask == "GM":
            print(f"[MASK] Using GM only (threshold {mask_threshold})")
            resampled_gm_mask = resample_to_img(gm_mask, func_img, interpolation="nearest")
            mask_img = math_img(f"gm >= {mask_threshold}", gm=resampled_gm_mask)

        elif mask == "BM":
            print(f"[MASK] Using brain mask")
            brain_resampled = resample_to_img(brain_img, func_img, interpolation='nearest')
            mask_img = brain_resampled    

        elif mask == "GMBM": 
            print(f"[MASK] Using GM ∩ Brain (default) (threshold {mask_threshold})")
            brain_resampled = resample_to_img(brain_img, func_img, interpolation='nearest')
            resampled_gm_mask = resample_to_img(gm_mask, func_img, interpolation="nearest")
            mask_img = math_img(f"(gm >= {mask_threshold}) & (brain > 0)", gm=resampled_gm_mask, brain=brain_resampled)

        elif mask == None:
            print(f"[MASK] no mask applied to data")
            mask_img = None
        else:
            raise ValueError(f"Unknown mask option '{mask}'. Supported options: GM, GMBM, None")

        # Create masker with final mask
        masker = get_masker(pilot=pilot, space=space, yeo_networks=yeo_networks, smoothing_fwhm=smoothing_fwhm, mask_img=mask_img)
        
        baseline_time_series = masker.fit_transform(baseline_clean_func_img, confounds=baseline_confounds)
        meditation_time_series = masker.fit_transform(meditation_clean_func_img, confounds=meditation_confounds)
        full_time_series = masker.fit_transform(full_clean_func_img, confounds=full_confounds)

    elif level == 'voxel':
        # Voxel-level time series

        # Extract, apply confounds and mask 
        # Load probabilistic GM mask and brain mask
        gm_prob_mask = load_img(GM_MASK)
        brain_img = load_img(os.path.join(PATH, BRAIN_MASK_PATHS[scan]))

        if mask == "GM":
            print(f"[MASK] Using GM voxel mask (threshold {mask_threshold})")
            gm_mask = math_img(f"img >= {mask_threshold}", img=gm_prob_mask)
            resampled_gm_mask = resample_to_img(gm_mask, func_img, interpolation="nearest")
            final_mask = resampled_gm_mask

        elif mask == "BM":
            print(f"[MASK] Using brain mask")
            brain_resampled = resample_to_img(brain_img, func_img, interpolation='nearest')
            final_mask = brain_resampled    

        elif mask == "GMBM": 
            print(f"[MASK] Using Gray Matter ∩ Brain Mask (threshold {mask_threshold})")
            gm_mask = math_img(f"img >= {mask_threshold}", img=gm_prob_mask)
            resampled_gm_mask = resample_to_img(gm_mask, func_img, interpolation="nearest")
            brain_resampled = resample_to_img(brain_img, func_img, interpolation='nearest')
            mask_img = math_img(f"(gm >= {mask_threshold}) & (brain > 0)", gm=resampled_gm_mask, brain=brain_resampled)
            final_mask = mask_img

        elif mask == None:
            print(f"[MASK] no mask applied to data")
            mask_img = None
        else:
            raise ValueError(f"Unknown mask option '{mask}'. Supported options: GM, GMBM, None")


        masked_data = apply_mask(func_img, final_mask)  # shape: (n_timepoints, n_voxels)      
        cleaned_data = clean(masked_data, confounds=full_confounds, detrend=True, standardize=True, smoothing_fwhm=smoothing_fwhm, low_pass=0.09, high_pass=0.006, t_r=2.2)
        
        # Step 3: Index time windows (2D indexing)
        baseline_time_series = cleaned_data[baseline_indices, :]     # shape: (N_baseline, V)
        meditation_time_series = cleaned_data[meditation_indices, :] # shape: (N_meditation, V)
        full_time_series = cleaned_data[full_indices, :]             # shape: (N_full, V)

    else:
         raise ValueError("level must be 'roi' or 'voxel'.")

    # Convert bad_indices to local indices
    bad_baseline_indices = np.where(np.isin(baseline_indices, bad_indices))[0]
    bad_meditation_indices = np.where(np.isin(meditation_indices, bad_indices))[0]
    bad_full_indices = np.where(np.isin(full_indices, bad_indices))[0]


    if fd_scrubbing == "interpol":
        baseline_time_series = interpolate_bad_volumes(baseline_time_series, bad_baseline_indices)
        meditation_time_series = interpolate_bad_volumes(meditation_time_series, bad_meditation_indices)
        full_time_series = interpolate_bad_volumes(full_time_series, bad_full_indices)

    elif fd_scrubbing is True:
        print(f"[SCRUB] {scan} Applying NaN scrubbing for high FD volumes (±1) ...")
        bad_baseline_indices_exp = expand_indices(bad_baseline_indices, baseline_time_series.shape[0])
        bad_meditation_indices_exp = expand_indices(bad_meditation_indices, meditation_time_series.shape[0])
        bad_full_indices_exp = expand_indices(bad_full_indices, full_time_series.shape[0])

        print(f"Scrubbing baseline: {len(bad_baseline_indices_exp)} timepoints")
        print(f"Scrubbing meditation: {len(bad_meditation_indices_exp)} timepoints")
        print(f"Scrubbing full: {len(bad_full_indices_exp)} timepoints")

        baseline_time_series = nan_scrub_timepoints(baseline_time_series, bad_baseline_indices_exp)
        meditation_time_series = nan_scrub_timepoints(meditation_time_series, bad_meditation_indices_exp)
        full_time_series = nan_scrub_timepoints(full_time_series, bad_full_indices_exp)

        print("NaN scrubbing applied successfully: high FD volumes (±1) replaced with NaN.")

    # Report summary of NaNs after scrubbing
    report_nan_stats(baseline_time_series, label="baseline")
    report_nan_stats(meditation_time_series, label="meditation")
    report_nan_stats(full_time_series, label="full")
    return baseline_time_series, meditation_time_series, full_time_series


def save_series(
    scan,
    space="native",
    fd_scrubbing=True,
    level='roi', 
    yeo_networks =7,    
    smoothing_fwhm=6,
    confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],        
    mask=None,
    mask_threshold=0.5,
    output_dir=OUTPUT_PATH
):
    """
    Extract and save baseline and meditation time series to pickle files.

    This function runs `get_series` to generate the cleaned time series data,
    then saves each condition (baseline and meditation) to separate pickle files.

    If a scan name contains 'MEICA' and confound_columns is not empty, 
    the function will skip this scan to avoid double regression.

    Parameters
    ----------
    scan : str
        Identifier for the specific scan (e.g., 'P1S2', 'P2S7MEICA').
    space : str
        "native" (default) or "MNI" – specifies Schaefer atlas space.
    fd_scrubbing : bool, optional
        If True, volumes with FD > FD_THRESHOLD are removed.
    confound_columns : list of str, optional
        Confound columns to be regressed out.
    level : {'roi', 'voxel'}, optional
        Granularity of the data. 'roi' (default) returns atlas-based ROI time series,
        'voxel' returns voxel-level time series.
    output_dir : str, optional
        Directory to store the output pickle files. Defaults to OUTPUT_PATH.
    """

    # 2) Generate the time series
    baseline_ts, meditation_ts, full_ts = get_series(
        scan=scan,
        space=space,
        fd_scrubbing=fd_scrubbing,
        level=level, 
        yeo_networks=yeo_networks,
        smoothing_fwhm=smoothing_fwhm,
        mask=mask,
        mask_threshold=mask_threshold,
        confound_columns=confound_columns
    )

    # 3) Create a short confound descriptor
    confound_string = short_confounds_name(confound_columns)
    mask_label = mask if mask is not None else "NOMASK"
    threshold_label = f"thr{mask_threshold}" if mask and "GM" in mask else ""


    # 4) Build the filenames
    if level == 'roi':
        level_name = f"roi-{yeo_networks}yeo_smooth-{smoothing_fwhm}"
    else:
        level_name = f"voxel_smooth-{smoothing_fwhm}"
    base_filename = f"{scan}-{space}_mask-{mask_label}{'_' + threshold_label if threshold_label else ''}_FD-{fd_scrubbing}-{FD_THRESHOLD}_conf-{confound_string}_lvl-{level_name}"
    
    # Define and create the subfolder inside OUTPUT_PATH
    os.makedirs(os.path.join(OUTPUT_PATH, "pkl_timeseries"), exist_ok=True)

    # complete filenames
    baseline_filename = os.path.join(output_dir, "pkl_timeseries", f"{base_filename}_baseline.pkl")
    meditation_filename = os.path.join(output_dir, "pkl_timeseries", f"{base_filename}_meditation.pkl")
    full_filename = os.path.join(output_dir, "pkl_timeseries", f"{base_filename}_full.pkl")

    # 5) Save to pickle
    os.makedirs(output_dir, exist_ok=True)
    with open(baseline_filename, 'wb') as bf:
        pickle.dump(baseline_ts, bf)
    with open(meditation_filename, 'wb') as mf:
        pickle.dump(meditation_ts, mf)
    with open(full_filename, 'wb') as ff:
        pickle.dump(full_ts, ff)
    
    print(f"[PARAMETERS] {scan} Level: {level}, Space: {space}, Mask: {mask_label} Smoothing: {smoothing_fwhm}mm, FD Scrubbing: {fd_scrubbing}, Confounds: {short_confounds_name(confound_columns)}")
    print(f"[SAVED] {scan} baseline, meditation and full time series saved to {output_dir}/pkl_timeseries/{base_filename}_baseline.pkl (, ...meditation.pkl, ...full.pkl)")

def load_series(
    scan,
    space="native",
    fd_scrubbing=True,
    confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],        
    level='roi', yeo_networks =7,
    smoothing_fwhm=6,
    mask=None,
    mask_threshold=0.5,
    output_dir=os.path.join(OUTPUT_PATH, "pkl_timeseries")
):
    """
    Load previously saved baseline and meditation time series from pickle files.

    Parameters
    ----------
    scan : str
        Identifier for the specific scan (e.g., 'P1S2', 'P2S7MEICA').
    fd_scrubbing : bool, optional
        If True, indicates FD scrubbing was used when saving the file.
    confound_columns : list of str, optional
        Confound columns that were regressed out when saving the file.
    level : {'roi', 'voxel'}, optional
        Granularity level used when the time series was saved ('roi' or 'voxel').
    output_dir : str, optional
        Directory containing the pickle files. Defaults to OUTPUT_PATH.

    Returns
    -------
    baseline_time_series : np.ndarray
        Baseline time series loaded from the pickle.
    meditation_time_series : np.ndarray
        Meditation time series loaded from the pickle.
    """
    # Use the same short confound descriptor
    confound_string = short_confounds_name(confound_columns)
    mask_label = mask if mask is not None else "NOMASK"
    threshold_label = f"thr{mask_threshold}" if mask and "GM" in mask else ""

# 
# 4) Build the filenames
    if level == 'roi':
        level_name = f"roi-{yeo_networks}yeo_smooth-{smoothing_fwhm}"
    else:
        level_name = f"voxel_smooth-{smoothing_fwhm}"
    base_filename = f"{scan}-{space}_mask-{mask_label}{'_' + threshold_label if threshold_label else ''}_FD-{fd_scrubbing}-{FD_THRESHOLD}_conf-{confound_string}_lvl-{level_name}"
    baseline_filename = os.path.join(output_dir, f"{base_filename}_baseline.pkl")
    meditation_filename = os.path.join(output_dir, f"{base_filename}_meditation.pkl")
    full_filename = os.path.join(output_dir, f"{base_filename}_full.pkl")


    if not os.path.exists(baseline_filename) or not os.path.exists(meditation_filename) or not os.path.exists(full_filename):
        raise FileNotFoundError(
            f"Pickle files not found for {scan}. "
            f"Expected: {baseline_filename} and {meditation_filename} and {full_filename}"
        )

    with open(baseline_filename, 'rb') as bf:
        baseline_time_series = pickle.load(bf)
    with open(meditation_filename, 'rb') as mf:
        meditation_time_series = pickle.load(mf)
    with open(full_filename, 'rb') as mf:
        full_time_series = pickle.load(mf)

    print(f"[LOAD] Baseline time series <- {baseline_filename}")
    print(f"[LOAD] Meditation time series <- {meditation_filename}")
    print(f"[LOAD] Full time series <- {full_filename}")
    print(f"Baseline shape: {baseline_time_series.shape} (level={level})")
    print(f"Meditation shape: {meditation_time_series.shape} (level={level})")
    print(f"Full shape: {full_time_series.shape} (level={level})")

    report_nan_stats(baseline_time_series, label=f"{scan} - baseline")
    report_nan_stats(meditation_time_series, label=f"{scan} - meditation")
    report_nan_stats(full_time_series, label=f"{scan} - full")


    return baseline_time_series, meditation_time_series, full_time_series


##############################################
###          4) MAIN EXECUTION            ###
##############################################

if __name__ == "__main__":
    # Example usage: 
    # Loop through all scans in P1_FUNC_IMG_PATHS and P2_FUNC_IMG_PATHS, 
    # save ROI-level and voxel-level series.
    all_scan_ids = list(P1_FUNC_IMG_PATHS.keys()) + list(P2_FUNC_IMG_PATHS.keys())
    print("Starting batch processing of scans...")

    for scan_id in all_scan_ids:
        print(f"\nProcessing scan: {scan_id}")
        print(f"Saving ROI-level series for {scan_id}...")
        save_series(scan=scan_id, level='roi', yeo_networks=7, output_dir=OUTPUT_PATH)
        save_series(scan=scan_id, level='roi', yeo_networks=17, output_dir=OUTPUT_PATH)

     #   print(f"Saving voxel-level series for {scan_id}...")
     #  save_series(scan=scan_id, level='voxel', output_dir=OUTPUT_PATH)

    print("\nAll scans processed and saved.")
