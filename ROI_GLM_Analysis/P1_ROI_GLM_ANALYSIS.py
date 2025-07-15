# Script for all ROI's in H1 (Ajna > YRS) on T1-weighted anatomy using custom design matrix
# Updated: Uses precomputed design matrix from build_design_matrix script

# %%
import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img, index_img
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from utils.config import (
    P1_PATH, P1_TR, P1_FUNC_IMG_PATHS_NATIVE,
    OUTPUT_PATH, P1_ANAT
)

# %%
# === Setup ===
working_dir = r"C:\Users\gaaya\OneDrive\Documents\GitHub\VajraCode-fMRI"
os.chdir(working_dir)
print(f"✅ Working directory set to: {os.getcwd()}")

# %%
scan_id = "P1S4"
meditation = "Ajna"
baseline = "YRS"
bold_path = os.path.join(P1_PATH, P1_FUNC_IMG_PATHS_NATIVE[scan_id])
bold_img = nib.load(bold_path)

# === Load precomputed design matrix ===
design_matrix_path = os.path.join(
    OUTPUT_PATH, "GLM_designs", "P1S4_mot6-wm-csf_glm_design_matrix.tsv"
)
design_matrix = pd.read_csv(design_matrix_path, sep="\t")

# === Truncate BOLD image to match design matrix length ===
bold_img_trimmed = index_img(bold_img, slice(0, design_matrix.shape[0]))

# === Extended ROI targets for H1 ===
rois = [
    "IFG", "DLPFC", "dACC", "Insula", "LeftThalamus",
    "FrontalMedialCortex", "TemporalGyri", "Hippocampus"
]

mask_base_path = os.path.join(
    r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois"
)

# %%
# === Run GLM per ROI ===
for roi in rois:
    print(f"\n===== Running GLM for {roi} ROI =====")

    mask_filename = f"{scan_id}_{roi}_mask_native.nii.gz"
    mask_path = os.path.join(mask_base_path, mask_filename)
    output_dir = os.path.join(OUTPUT_PATH, "ROI_GLM", scan_id, roi)
    os.makedirs(output_dir, exist_ok=True)

    try:
        mask_img = nib.load(mask_path)
        mask_img = resample_to_img(mask_img, bold_img_trimmed, interpolation='nearest')
    except FileNotFoundError:
        print(f"⚠️ Mask not found: {mask_path}")
        continue

    glm = FirstLevelModel(
        t_r=P1_TR,
        mask_img=mask_img,
        slice_time_ref=0.5,
        minimize_memory=False
    )

    glm = glm.fit(bold_img_trimmed, design_matrices=design_matrix)
    contrast = glm.compute_contrast("Ajna - YRS", output_type="effect_size")

    # Save beta data
    roi_beta = contrast.get_fdata()
    mean_beta = np.mean(roi_beta[roi_beta != 0])
    np.savetxt(os.path.join(output_dir, f"{scan_id}_{roi}_mean_beta.txt"), [mean_beta])
    nib.save(contrast, os.path.join(output_dir, f"{scan_id}_{roi}_beta_map.nii.gz"))
    print(f"✅ Mean beta (Ajna > YRS) in {roi}: {mean_beta:.4f}")

    # Plot contrast
    cut_coords = find_xyz_cut_coords(contrast)
    plot_stat_map(
        contrast,
        bg_img=P1_ANAT,
        display_mode='ortho',
        cut_coords=cut_coords,
        threshold=0.5,
        title=f"Pilot 1 {roi} Ajna > YRS",
        colorbar=True,
        draw_cross=False
    )
    plt.savefig(os.path.join(output_dir, f"{scan_id}_{roi}_contrast_T1w.png"), dpi=300)


# %%
# === Setup ===
working_dir = r"C:\Users\gaaya\OneDrive\Documents\GitHub\VajraCode-fMRI"
os.chdir(working_dir)
print(f"✅ Working directory set to: {os.getcwd()}")
from nilearn.glm import threshold_stats_img
from nilearn.input_data import NiftiMasker
from scipy.stats import t

# %%
# === Scan and Design ===
scan_id = "P1S4"
meditation = "Ajna"
baseline   = "YRS"

bold_path = os.path.join(P1_PATH, P1_FUNC_IMG_PATHS_NATIVE[scan_id])
bold_img  = nib.load(bold_path)

# Load design matrix
design_matrix_path = os.path.join(
    OUTPUT_PATH, "GLM_designs", f"{scan_id}_mot6-wm-csf_glm_design_matrix.tsv"
)
design_matrix = pd.read_csv(design_matrix_path, sep="\t")

# Trim BOLD to match design
bold_img_trimmed = index_img(bold_img, slice(0, design_matrix.shape[0]))

# %%
# === ROI List & Mask Base ===
rois = [
    "IFG", "DLPFC", "dACC", "Insula", "LeftThalamus",
    "FrontalMedialCortex", "TemporalGyri", "Hippocampus"
]
mask_base = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois"

# Storage for ROI-level inference
df_inference = []
df_correction = []

# %%
# === Main Loop ===
for roi in rois:
    print(f"\n===== ROI: {roi} =====")
    mask_file = os.path.join(mask_base, f"{scan_id}_{roi}_mask_native.nii.gz")
    output_dir = os.path.join(OUTPUT_PATH, "ROI_GLM_FWE", scan_id, roi)
    os.makedirs(output_dir, exist_ok=True)

    # Load & resample mask
    try:
        mask_img = nib.load(mask_file)
        mask_img = resample_to_img(mask_img, bold_img_trimmed, interpolation='nearest')
    except FileNotFoundError:
        print(f"⚠️ Mask not found: {mask_file}")
        continue

    # Fit GLM restricted to ROI
    glm = FirstLevelModel(
        t_r=P1_TR,
        mask_img=mask_img,
        slice_time_ref=0.5,
        minimize_memory=False
    ).fit(bold_img_trimmed, design_matrices=design_matrix)

    # Compute beta and t maps
    beta_map = glm.compute_contrast(f"{meditation} - {baseline}", output_type="effect_size")
    t_map    = glm.compute_contrast(f"{meditation} - {baseline}", output_type="stat")

    # Small-volume Bonferroni correction within ROI
    thr_t_map, t_thresh = threshold_stats_img(
        t_map,
        alpha=0.05,
        height_control='bonferroni',
        mask_img=mask_img
    )
    print(f"  → Bonferroni threshold t ≥ {t_thresh:.2f}")

    # Save maps
    nib.save(beta_map,  os.path.join(output_dir, f"{scan_id}_{roi}_beta_FWE.nii.gz"))
    nib.save(t_map,     os.path.join(output_dir, f"{scan_id}_{roi}_tmap.nii.gz"))
    nib.save(thr_t_map, os.path.join(output_dir, f"{scan_id}_{roi}_tmap_FWE.nii.gz"))

    # ROI-level inference (mean time series)
    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    Y       = masker.fit_transform(bold_img_trimmed)
    y       = Y.mean(axis=1)
    X       = design_matrix.values
    pinvX   = np.linalg.pinv(X)
    beta_hat= pinvX.dot(y)
    y_hat   = X.dot(beta_hat)
    resid   = y - y_hat
    dof     = y.shape[0] - np.linalg.matrix_rank(X)
    # contrast vector
    v = np.zeros(X.shape[1])
    v[design_matrix.columns.get_loc(meditation)] = 1
    v[design_matrix.columns.get_loc(baseline)]   = -1
    effect  = v.dot(beta_hat)
    mse     = (resid**2).sum() / dof
    var_eff = mse * v.dot(pinvX.dot(pinvX.T).dot(v))
    t_obs   = effect / np.sqrt(var_eff)
    p_val   = 2 * (1 - t.cdf(abs(t_obs), dof))
    df_inference.append({
        'ROI': roi,
        'mean_beta': round(effect,4),
        't_obs': round(t_obs,4),
        'df': int(dof),
        'p_val': round(p_val,4)
    })

    # Voxelwise correction summary (Bonferroni)
    t_data = t_map.get_fdata()
    mask_data = mask_img.get_fdata() > 0
    nvox = mask_data.sum()
    if nvox > 0:
        max_t = float(np.max(np.abs(t_data[mask_data])))
        p_unc = 2 * (1 - t.cdf(max_t, dof))
        p_bonf= min(1.0, p_unc * nvox)
    else:
        max_t, p_unc, p_bonf = np.nan, np.nan, np.nan
    df_correction.append({
        'ROI': roi,
        'n_voxels': int(nvox),
        'max_t': round(max_t,4),
        'p_uncorrected': round(p_unc,4),
        'p_bonferroni': round(p_bonf,4)
    })

    # Plot and save
    cut_coords = find_xyz_cut_coords(t_map)
    for img, title, fname, thr in [
        (t_map,    f"{roi} t-map (unc)",      f"tmap_unc_{roi}.png",       0),
        (thr_t_map,f"{roi} t-map (FWE bonf)", f"tmap_FWE_{roi}.png",  t_thresh)
    ]:
        disp = plot_stat_map(
            img, bg_img=P1_ANAT, display_mode='ortho',
            cut_coords=cut_coords, threshold=thr,
            title=title, colorbar=True, draw_cross=False
        )
        disp.frame_axes.figure.savefig(
            os.path.join(output_dir, fname), dpi=300
        )
        plt.close()

# %%
# === Display ROI-Level Contrast Inference ===
print("\n📊 ROI-Level Contrast Inference")
print(pd.DataFrame(df_inference).to_string(index=False))



# %%
