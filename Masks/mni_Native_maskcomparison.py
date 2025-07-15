# This is the script to compare MNI and native masks for the CHAKRA project.

# %%
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Define output directory
output_base = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\maskcomparisons"
os.makedirs(output_base, exist_ok=True)

# Define mask pairs (MNI, Native, Region Name, Pilot Label)
mask_pairs = [
    # Pilot 1
    ("HarvardOxford_IFG_mask_MNI.nii.gz", "P1S4_IFG_mask_native.nii.gz", "IFG", "Pilot 1"),
    ("HarvardOxford_DLPFC_mask_MNI.nii.gz", "P1S4_DLPFC_mask_native.nii.gz", "DLPFC", "Pilot 1"),
    ("HarvardOxford_dACC_mask_MNI.nii.gz", "P1S4_dACC_mask_native.nii.gz", "dACC", "Pilot 1"),
    ("HarvardOxford_Insula_mask_MNI.nii.gz", "P1S4_Insula_mask_native.nii.gz", "Insula", "Pilot 1"),
    ("HarvardOxford_LeftThalamus_mask_MNI.nii.gz", "P1S4_LeftThalamus_mask_native.nii.gz", "LeftThalamus", "Pilot 1"),
    ("HarvardOxford_FrontalMedialCortex_mask_MNI.nii.gz", "P1S4_FrontalMedialCortex_mask_native.nii.gz", "FrontalMedialCortex", "Pilot 1"),
    ("HarvardOxford_TemporalGyri_mask_MNI.nii.gz", "P1S4_TemporalGyri_mask_native.nii.gz", "TemporalGyri", "Pilot 1"),
    ("HarvardOxford_Hippocampus_mask_MNI.nii.gz", "P1S4_Hippocampus_mask_native.nii.gz", "Hippocampus", "Pilot 1"),

    # Pilot 2
    ("HarvardOxford_IFG_mask_MNI.nii.gz", "P2S8MEICA_IFG_mask_native.nii.gz", "IFG", "Pilot 2"),
    ("HarvardOxford_DLPFC_mask_MNI.nii.gz", "P2S8MEICA_DLPFC_mask_native.nii.gz", "DLPFC", "Pilot 2"),
    ("HarvardOxford_dACC_mask_MNI.nii.gz", "P2S8MEICA_dACC_mask_native.nii.gz", "dACC", "Pilot 2"),
    ("HarvardOxford_Insula_mask_MNI.nii.gz", "P2S8MEICA_Insula_mask_native.nii.gz", "Insula", "Pilot 2"),
    ("HarvardOxford_LeftThalamus_mask_MNI.nii.gz", "P2S8MEICA_LeftThalamus_mask_native.nii.gz", "LeftThalamus", "Pilot 2"),
    ("HarvardOxford_FrontalMedialCortex_mask_MNI.nii.gz", "P2S8_FrontalMedialCortex_mask_native.nii.gz", "FrontalMedialCortex", "Pilot 2"),
    ("HarvardOxford_TemporalGyri_mask_MNI.nii.gz", "P2S8_TemporalGyri_mask_native.nii.gz", "TemporalGyri", "Pilot 2"),
    ("HarvardOxford_Hippocampus_mask_MNI.nii.gz", "P2S8_Hippocampus_mask_native.nii.gz", "Hippocampus", "Pilot 2"),
]

# Base mask directory
mask_dir = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI\output\native_rois"

# Generate comparison plots
for mni_file, native_file, region, pilot in mask_pairs:
    mni_path = os.path.join(mask_dir, mni_file)
    native_path = os.path.join(mask_dir, native_file)

    mni_img = nib.load(mni_path)
    native_img = nib.load(native_path)

    mni_data = mni_img.get_fdata()
    native_data = native_img.get_fdata()

    mni_slice = np.argmax(np.sum(mni_data, axis=(0, 1)))
    native_slice = np.argmax(np.sum(native_data, axis=(0, 1)))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"{region} MNI vs Native for {pilot} - Axial Slice", fontsize=12)

    axs[0].imshow(mni_data[:, :, mni_slice], cmap="gray")
    axs[0].set_title(f"{region} MNI mask - {pilot} (axial)")
    axs[0].axis("off")

    axs[1].imshow(native_data[:, :, native_slice], cmap="gray")
    axs[1].set_title(f"{region} Native mask - {pilot} (axial)")
    axs[1].axis("off")

    plt.tight_layout(pad=1.0)

    save_dir = os.path.join(output_base, pilot.replace(" ", ""))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{region}_{pilot.replace(' ', '')}_comparison.png"), dpi=300)
    plt.close()

print("✅ All MNI vs Native mask comparisons saved.") 






# %%
