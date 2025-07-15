# %%
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import datasets

# === Output Path ===
output_dir = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "HarvardOxford_TemporalGyri_mask_MNI.nii.gz")

# === Load Harvard-Oxford Cortical Atlas with 0% threshold to retain all voxels ===
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split=True)
atlas_img = atlas.maps
labels = atlas.labels
atlas_data = atlas_img.get_fdata()

# === Define labels of interest ===
target_labels = [
    'Superior Temporal Gyrus, posterior division',
    'Middle Temporal Gyrus, posterior division',
    'Middle Temporal Gyrus, temporooccipital part'
]

# === Find label indices ===
label_indices = [i for i, label in enumerate(labels) if any(tl in label for tl in target_labels)]
print("✅ Temporal gyri label indices found:", label_indices)

# === Create binary mask ===
mask_data = np.isin(atlas_data, label_indices).astype(np.uint8)
mask_img = nib.Nifti1Image(mask_data, affine=atlas_img.affine)
nib.save(mask_img, output_path)
print(f"✅ Temporal Gyri MNI mask saved to: {output_path}")

# === Visualize max intensity slice ===
central_slice = np.argmax(np.sum(mask_data, axis=(0, 1)))
plt.imshow(mask_data[:, :, central_slice], cmap='gray')
plt.title("TemporalGyri MNI mask – max intensity axial slice")
plt.colorbar()
plt.show()

# %%
