# %%
import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_roi
from utils.config import P2_ANAT

# === Path to IFG native-space mask for Pilot 2 ===
ifg_mask_path = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois\P2S8_IFG_mask_native.nii.gz"

# === Load images ===
anat_img = nib.load(P2_ANAT)
mask_img = nib.load(ifg_mask_path)

# === Plot ROI on T1w ===
plot_roi(
    mask_img,
    bg_img=anat_img,
    display_mode='ortho',
    title="P2 IFG Mask on T1-weighted Anatomy",
    draw_cross=False,
    colorbar=True
)
plt.show()
# %%
