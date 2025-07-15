# %%
import os
import numpy as np
import nibabel as nib
from nilearn import datasets
import matplotlib.pyplot as plt

# === Set working directory / output path ===
output_dir = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "HarvardOxford_IFG_mask_MNI.nii.gz")

# === Load Harvard-Oxford Atlas (symmetric, 2mm, 25% threshold) ===
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
atlas_img = atlas.maps
labels = atlas.labels
atlas_data = atlas_img.get_fdata()

# === IFG subregions based on available label names ===
# Your screenshots show the following IFG-related regions are valid:
ifg_labels = [
    'Left Inferior Frontal Gyrus, pars triangularis',
    'Right Inferior Frontal Gyrus, pars triangularis',
    'Left Inferior Frontal Gyrus, pars opercularis',
    'Right Inferior Frontal Gyrus, pars opercularis'
]

# Match these labels to the index numbers in the atlas
ifg_indices = [i for i, label in enumerate(labels) if label in ifg_labels]
print("✅ IFG label indices found:", ifg_indices)

# === Create mask ===
mask_data = np.isin(atlas_data, ifg_indices).astype(np.uint8)
mask_img = nib.Nifti1Image(mask_data, affine=atlas_img.affine)
nib.save(mask_img, output_path)
print(f"✅ IFG MNI mask saved to: {output_path}")

# === Quality check: print voxel count and visualize ===
nonzero_voxels = np.count_nonzero(mask_data)
print("🔍 Non-zero voxels in IFG MNI mask:", nonzero_voxels)

# Optional: plot middle axial slice
plt.imshow(mask_data[:, :, mask_data.shape[2] // 2], cmap="gray")
plt.title("Middle axial slice of IFG MNI mask")
plt.colorbar()
plt.show()

# %%
mask_path = r"C:/Users/gaaya/OneDrive/Desktop/UVA Thesis stuff/CHAKRA_PREP/GitHub/ChakrafMRI/output/native_rois/P1S4_IFG_mask_native.nii.gz"

# === Load mask ===
img = nib.load(mask_path)
data = img.get_fdata()

# === Count non-zero voxels ===
nonzero_voxels = np.count_nonzero(data)
print(f"🔎 Non-zero voxels in native IFG mask: {nonzero_voxels}")

# === Visualize a central axial slice ===
mid_slice = data.shape[2] // 2
plt.imshow(data[:, :, mid_slice], cmap='gray')
plt.title("Middle axial slice of native-space IFG mask")
plt.colorbar()
plt.axis("off")
plt.show()
# %%
