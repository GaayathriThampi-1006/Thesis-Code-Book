# %%

import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_roi
from utils.config import P1_ANAT, P2_ANAT  # Load anatomical paths from config

# === Directories ===
base_dir = r"C:\Users\gaaya\OneDrive\Desktop\UVA Thesis stuff\CHAKRA_PREP\GitHub\ChakrafMRI"
native_rois = os.path.join(base_dir, "output", "native_rois")
overlay_output = os.path.join(base_dir, "output", "maskoverlays")
os.makedirs(overlay_output, exist_ok=True)

# === Define native mask files per pilot ===
pilot_masks = {
    "Pilot1": {
        "anatomical": P1_ANAT,
        "masks": {
            "IFG": "P1S4_IFG_mask_native.nii.gz",
            "DLPFC": "P1S4_DLPFC_mask_native.nii.gz",
            "dACC": "P1S4_dACC_mask_native.nii.gz",
            "Insula": "P1S4_Insula_mask_native.nii.gz",
            "LeftThalamus": "P1S4_LeftThalamus_mask_native.nii.gz",
            "FrontalMedialCortex": "P1S4_FrontalMedialCortex_mask_native.nii.gz",
            "TemporalGyri": "P1S4_TemporalGyri_mask_native.nii.gz",
            "Hippocampus": "P1S4_Hippocampus_mask_native.nii.gz"
        }
    },
    "Pilot2": {
        "anatomical": P2_ANAT,
        "masks": {
            "IFG": "P2S8MEICA_IFG_mask_native.nii.gz",
            "DLPFC": "P2S8MEICA_DLPFC_mask_native.nii.gz",
            "dACC": "P2S8MEICA_dACC_mask_native.nii.gz",
            "Insula": "P2S8MEICA_Insula_mask_native.nii.gz",
            "LeftThalamus": "P2S8MEICA_LeftThalamus_mask_native.nii.gz",
            "FrontalMedialCortex": "P2S8_FrontalMedialCortex_mask_native.nii.gz",
            "TemporalGyri": "P2S8_TemporalGyri_mask_native.nii.gz",
            "Hippocampus": "P2S8_Hippocampus_mask_native.nii.gz"
        }
    }
}

# === Loop through and plot overlays ===
for pilot, info in pilot_masks.items():
    anat = nib.load(info["anatomical"])
    pilot_dir = os.path.join(overlay_output, pilot)
    os.makedirs(pilot_dir, exist_ok=True)

    for roi_name, mask_filename in info["masks"].items():
        mask_path = os.path.join(native_rois, mask_filename)
        try:
            mask_img = nib.load(mask_path)
        except FileNotFoundError:
            print(f"⚠️ Could not find: {mask_path}")
            continue

        print(f"🧠 Overlaying {roi_name} on T1w for {pilot}...")

        fig = plot_roi(
            roi_img=mask_img,
            bg_img=anat,
            display_mode="ortho",
            title=f"{roi_name} mask on T1w - {pilot}",
            draw_cross=False
        )
        fig_path = os.path.join(pilot_dir, f"{roi_name}_overlay.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"✅ Saved overlay to: {fig_path}")

# %%
