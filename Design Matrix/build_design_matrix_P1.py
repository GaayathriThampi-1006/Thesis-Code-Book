def main(scan_id="P1S4"):
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.plotting import plot_design_matrix
    from utils.config import (
        OUTPUT_PATH, FD_THRESHOLD,
        P1_TR, P1_TIMINGS, P1_VOLNUM, P1_CONFOUNDS, P1_PATH,
        P2_TR, P2_TIMINGS, P2_VOLNUM, P2_CONFOUNDS, P2_PATH
    )
    from utils.data import seconds_to_TR_indices, expand_indices, short_confounds_name
    from sklearn.preprocessing import StandardScaler

    # Set meditation type dynamically
    if scan_id == "P1S5":
        meditation = "Anahata"
    elif scan_id == "P1S4":
        meditation = "Ajna"
    else:
        raise ValueError(f"Unknown scan ID {scan_id} – please provide a supported Ajna or Anahata scan")

    print(f"🔧 Building GLM for {scan_id} ({meditation})...")


    output_dir = os.path.join(OUTPUT_PATH, "GLM_designs")
    os.makedirs(output_dir, exist_ok=True)

    # GLM design matrix parameters
    drift_model = 'polynomial'  #  
    drift_order = 2             # 2nd-order polynomial drift
    high_pass = None            # High-pass filter cutoff (Hz) - not apllied in polynomial drift models, redundant
    hrf_model= None 

    # === Dynamically load scan-specific parameters from config ===
    if scan_id.startswith("P1"):
        PATH = P1_PATH
        num_vols = P1_VOLNUM[scan_id]
        confounds_path = PATH + P1_CONFOUNDS[scan_id]
        timings = P1_TIMINGS[scan_id]
        TR = P1_TR
    elif scan_id.startswith("P2"):
        PATH = P2_PATH
        num_vols = P2_VOLNUM[scan_id]
        confounds_path = PATH + P2_CONFOUNDS[scan_id]
        timings = P2_TIMINGS[scan_id]
        TR = P2_TR
    else:
        raise ValueError(f"Unrecognized scan prefix in {scan_id}")
    print(f"\nTotal number of volumes for {scan_id}: {num_vols}\n")

    # === Load confounds and identify high-motion TRs for scrubbing ===
    confounds_df = pd.read_csv(confounds_path, sep='\t')
    fd = confounds_df['framewise_displacement'].fillna(0).values
    high_fd_TRs = np.where(fd > FD_THRESHOLD)[0]
    scrubbed_TRs = set(expand_indices(high_fd_TRs, max_len=len(fd)))

    # === Define task-relevant TRs for YRS and Anahata ===
    yrs_start, yrs_end = seconds_to_TR_indices(timings['YRS']['start'], timings['YRS']['end'], TR)
    ana_start, ana_end = seconds_to_TR_indices(timings[meditation]['start'], timings[meditation]['end'], TR)
    task_TRs = set(range(yrs_start, yrs_end + 1)).union(range(ana_start, ana_end + 1))

    # Restrict scrubbing to task-relevant TRs only
    scrubbed_TRs = scrubbed_TRs.intersection(task_TRs)

    # === Final TRs per condition ===
    yrs_TRs = set(range(yrs_start, yrs_end + 1))
    ana_TRs = set(range(ana_start, ana_end + 1))


    # === Create spike regressors for scrubbed TRs ===
    glm_n_volumes = ana_end + 1  # Stop GLM at end of Anahata condition
    spike_regressors = pd.DataFrame(
        np.zeros((glm_n_volumes, len(scrubbed_TRs))),
        columns=[f"scrub_{tr}" for tr in sorted(scrubbed_TRs)]
    )

    for i, tr in enumerate(sorted(scrubbed_TRs)):
        if tr < glm_n_volumes:  # Only include if within modeled time
            spike_regressors.iloc[tr, i] = 1

    # === Initialize design matrix frame with zeroed boxcars ===
    frame_times = np.arange(glm_n_volumes) * TR
    conditions = pd.DataFrame({'frame_times': frame_times})
    conditions['YRS'] = 0
    conditions[meditation] = 0

    # Populate condition regressors (sustained boxcars)
    conditions.loc[list(yrs_TRs), 'YRS'] = 1
    conditions.loc[list(ana_TRs), meditation] = 1

    # === Select confound regressors to include (standard motion + WM + CSF) ===
    confounds_matrix = confounds_df[[
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z",
        "white_matter", "csf"
    ]].iloc[:glm_n_volumes]
    confound_str = short_confounds_name(confounds_matrix.columns.tolist())


    # Standardize confounds (only confounds, not condition boxcars)
    scaler = StandardScaler()
    confounds_scaled = pd.DataFrame(
        scaler.fit_transform(confounds_matrix),
        columns=confounds_matrix.columns
    )


    # === Construct GLM Design Matrix using Nilearn ===
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=None,
        add_regs=pd.concat([conditions[['YRS', meditation]], 
        confounds_scaled, 
        spike_regressors
        ], 
        axis=1).values,
        add_reg_names=['YRS', meditation] 
        + confounds_matrix.columns.tolist() 
        + spike_regressors.columns.tolist(),
        hrf_model=hrf_model,
        drift_model=drift_model,
        drift_order=drift_order,
        high_pass=high_pass,
    )

    # === Save design matrix and plot ===
    design_matrix.to_csv(os.path.join(output_dir, f"{scan_id}_{confound_str}_glm_design_matrix.tsv"), sep="\t")

    # 💡 Inspect condition number here
    cond_number = np.linalg.cond(design_matrix.values)
    print(f"🔍 Condition number of design matrix: {cond_number:.2f}")
    if cond_number > 1000:
        print("⚠️ Warning: Design matrix is poorly conditioned — check for collinearity!")

    plot_design_matrix(design_matrix, rescale=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{scan_id}_{confound_str}_design_matrix.png"), dpi=300)
    plt.close()
    plt.show

    print(f"✅ Saved design matrix and design matrix plot to {output_dir}")


# === ===
if __name__ == "__main__":
    main("P1S4")  