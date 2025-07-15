# Updated P2 ROI GLM Analysis using precomputed design matrix and truncated BOLD

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
    P2_PATH, P2_TR, P2_MEICA_FUNC_IMG_PATHS_NATIVE,
    P2_TIMINGS, OUTPUT_PATH, P2_ANAT
)

# %%
# === Setup ===
working_dir = r"C:\Users\gaaya\OneDrive\Documents\GitHub\VajraCode-fMRI"
os.chdir(working_dir)

# %%
scan_id = "P2S8MEICA"
meditation = "Ajna"
baseline = "YRS"
bold_path = os.path.join(P2_PATH, P2_MEICA_FUNC_IMG_PATHS_NATIVE[scan_id])
bold_img = nib.load(bold_path)

# === Load precomputed design matrix ===
design_matrix_path = os.path.join(
    OUTPUT_PATH, "GLM_designs", "P2S8MEICA_none_glm_design_matrix.tsv"
)
design_matrix = pd.read_csv(design_matrix_path, sep="\t")

# === Truncate BOLD image to match design matrix length ===
bold_img_trimmed = index_img(bold_img, slice(0, design_matrix.shape[0]))

# === ROI targets ===
rois = [
    "IFG", "DLPFC", "dACC", "Insula", "LeftThalamus",
    "FrontalMedialCortex", "TemporalGyri", "Hippocampus"
]
mask_base_path = r"C:/Users/gaaya/OneDrive/Desktop/UVA Thesis stuff/CHAKRA_PREP/GitHub/ChakrafMRI/output/native_rois"

# %%
# === Run GLM per ROI ===
for roi in rois:
    print(f"\n===== Running GLM for {roi} ROI =====")

    mask_filename = f"{scan_id}_{roi}_mask_native.nii.gz"
    mask_path = os.path.join(mask_base_path, mask_filename)
    output_dir = os.path.join(OUTPUT_PATH, "ROI_GLM", scan_id, roi)
    os.makedirs(output_dir, exist_ok=True)

    mask_img = nib.load(mask_path)
    mask_img = resample_to_img(mask_img, bold_img_trimmed, interpolation='nearest')

    glm = FirstLevelModel(
        t_r=P2_TR,
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

    cut_coords = find_xyz_cut_coords(contrast)
    plot_stat_map(
        contrast,
        bg_img=P2_ANAT,
        display_mode='ortho',
        cut_coords=cut_coords,
        threshold=0.5,
        title=f"Pilot 2 {roi} Ajna > YRS",
        colorbar=True,
        draw_cross=False
    )
    plt.savefig(os.path.join(output_dir, f"{scan_id}_{roi}_contrast_T1w.png"), dpi=300)
    plt.show()


# %%  
# === ROI‐Level Contrast Inference (mean time series) ===
from nilearn.input_data import NiftiMasker
from scipy.stats import t

df_inference = []

for roi in rois:
    # Load & resample the ROI mask exactly as above
    mask_file = os.path.join(mask_base_path, f"{scan_id}_{roi}_mask_native.nii.gz")
    mask_img  = nib.load(mask_file)
    mask_img  = resample_to_img(mask_img, bold_img_trimmed, interpolation='nearest')

    # 1) Extract the ROI‐mean time series
    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    Y      = masker.fit_transform(bold_img_trimmed)  # shape (T, V)
    y      = Y.mean(axis=1)                          # shape (T,)

    # 2) Build & fit the univariate GLM
    X       = design_matrix.values                   # (T × p)
    pinvX   = np.linalg.pinv(X)
    beta_hat= pinvX.dot(y)                           # (p,)
    y_hat   = X.dot(beta_hat)
    resid   = y - y_hat
    dof     = y.shape[0] - np.linalg.matrix_rank(X)

    # 3) Define the Ajna‐minus‐YRS contrast
    v = np.zeros(X.shape[1])
    v[design_matrix.columns.get_loc(meditation)] = 1
    v[design_matrix.columns.get_loc(baseline)]   = -1

    # 4) Compute effect, t‐statistic, and p‐value
    effect  = v.dot(beta_hat)
    mse     = (resid**2).sum() / dof
    var_eff = mse * (v.dot(pinvX.dot(pinvX.T).dot(v)))
    t_obs   = effect / np.sqrt(var_eff)
    p_val   = 2 * (1 - t.cdf(abs(t_obs), dof))


    # 5) Store results
    df_inference.append({
        'ROI':       roi,
        'mean_beta': round(effect,4),
        't_obs':     round(t_obs,4),
        'df':        int(dof),
        'p_val':     round(p_val,4)
    })

# 6) Print the inference table
print("ROI‐Level Contrast Inference")
print(pd.DataFrame(df_inference).to_string(index=False))



# %%
