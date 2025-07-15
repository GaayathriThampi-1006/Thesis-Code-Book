# THIS IS FOR THE INSULA

# %%
import os
import numpy as np
import nibabel as nib
from nilearn import datasets
import matplotlib.pyplot as plt

# === Set working directory / output path ===
output_dir = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "HarvardOxford_Insula_mask_MNI.nii.gz")

# === Load Harvard-Oxford Atlas (symmetric, 2mm, 25% threshold) ===
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
atlas_img = atlas.maps
labels = atlas.labels
atlas_data = atlas_img.get_fdata()

# === Insula labels from Harvard-Oxford Atlas ===
insula_labels = [
    'Left Insular Cortex',
    'Right Insular Cortex'
]

# === Match labels to indices ===
insula_indices = [i for i, label in enumerate(labels) if label in insula_labels]
print("✅ Insula label indices found:", insula_indices)

# === Create binary mask ===
mask_data = np.isin(atlas_data, insula_indices).astype(np.uint8)
mask_img = nib.Nifti1Image(mask_data, affine=atlas_img.affine)
nib.save(mask_img, output_path)
print(f"✅ Insula MNI mask saved to: {output_path}")

# === Sanity check: count nonzero voxels ===
nonzero_voxels = np.count_nonzero(mask_data)
print("🔍 Non-zero voxels in Insula MNI mask:", nonzero_voxels)

# === Optional: visualize middle slice ===
plt.imshow(mask_data[:, :, mask_data.shape[2] // 2], cmap="gray")
plt.title("Middle axial slice of Insula MNI mask")
plt.colorbar()
plt.show()

central_slice = np.argmax(np.sum(mask_data, axis=(0, 1)))
plt.imshow(mask_data[:, :, central_slice], cmap='gray')
plt.title("Max intensity axial slice of Insula mask")
plt.colorbar()
plt.show()

# %%

os.chdir(r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois")

# Load the native-space Insula mask
img = nib.load("P1S4_Insula_mask_native.nii.gz")
data = img.get_fdata()

# Count nonzero voxels
nonzero_voxels = np.count_nonzero(data)
print(f"🔍 Non-zero voxels in native Insula mask: {nonzero_voxels}")

# Find the slice with max intensity across axial plane
central_slice = np.argmax(np.sum(data, axis=(0, 1)))

# Plot that slice
plt.imshow(data[:, :, central_slice], cmap='gray')
plt.title("Max intensity axial slice of Insula native-space mask")
plt.colorbar()
plt.show()

# %%
