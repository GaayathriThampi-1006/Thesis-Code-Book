# %%
# %%
from nilearn import datasets
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# === Set correct output directory and filename ===
output_dir = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois"
output_path = os.path.join(output_dir, "HarvardOxford_FrontalMedialCortex_mask_MNI.nii.gz")

# === Load full HO cortical atlas (no thresholding) ===
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split=True)
atlas_img = atlas.maps
labels = atlas.labels
atlas_data = atlas_img.get_fdata()

# === Get index of "Frontal Medial Cortex" ===
target_label = "Frontal Medial Cortex"
label_index = [i for i, label in enumerate(labels) if target_label in label]
print("✅ Matched label index:", label_index)

# === Create binary mask ===
mask_data = np.isin(atlas_data, label_index).astype(np.uint8)
mask_img = nib.Nifti1Image(mask_data, affine=atlas_img.affine)

# === Save to correct path ===
nib.save(mask_img, output_path)
print(f"✅ Saved MNI mask to: {output_path}")

# === Sanity check plot ===
central_slice = np.argmax(np.sum(mask_data, axis=(0, 1)))
plt.imshow(mask_data[:, :, central_slice], cmap='gray')
plt.title("FrontalMedialCortex MNI mask - max intensity axial slice")
plt.colorbar()
plt.show()

# %%
