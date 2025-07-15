def main(scan_id="P1S8MEICA"):


    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.plotting import plot_design_matrix
    from utils.config import (OUTPUT_PATH, FD_THRESHOLD, P2_TR, P2_TIMINGS, P2_VOLNUM, P2_CONFOUNDS, P2_PATH)
    from utils.data import seconds_to_TR_indices, expand_indices, short_confounds_name
    from sklearn.preprocessing import StandardScaler
    from itertools import groupby
    from operator import itemgetter


    # Set meditation type dynamically
    if scan_id == "P2S7MEICA":
        meditation = "Anahata"
    elif scan_id == "P2S8MEICA":
        meditation = "Ajna"
    else:
        raise ValueError(f"Unknown scan ID {scan_id} – please provide a supported Ajna or Anahata scan")

    print(f"🔧 Building GLM for {scan_id} ({meditation})...")
 
    output_dir = os.path.join(OUTPUT_PATH, "GLM_designs")
    os.makedirs(output_dir, exist_ok=True)

    # GLM design matrix parameters
    drift_model = None          # taken care of by tedana ICA denoising
    high_pass = None            # same as above
    low_pass = None             # same as above (not applied anywhere in this script)
    hrf_model = None


    PATH = P2_PATH
    TR = P2_TR
    if scan_id == "P2S7MEICA":
        num_vols = P2_VOLNUM["P2S7ME"]
        confounds_path = PATH + P2_CONFOUNDS["P2S7ME"]
        timings = P2_TIMINGS["P2S7ME"]
    elif scan_id == "P2S8MEICA":
        num_vols = P2_VOLNUM["P2S8ME"]
        confounds_path = PATH + P2_CONFOUNDS["P2S8ME"]
        timings = P2_TIMINGS["P2S8ME"]
    else:
        raise ValueError(f"Unknown scan ID {scan_id} – please provide a supported Ajna or Anahata scan")

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

    # === Select confound regressors to include (None due to tedana ICA denoising) ===
    confounds = []
    confound_str = short_confounds_name(confounds)

    # === Construct GLM Design Matrix using Nilearn ===
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=None,
        add_regs=pd.concat([conditions[['YRS', meditation]], 
        spike_regressors], 
        axis=1).values,
        add_reg_names=['YRS', meditation] 
        + spike_regressors.columns.tolist(),
        hrf_model=hrf_model,
        drift_model=drift_model,
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

    print(f"✅ Saved design matrix and design matrix plot to {output_dir}")


# === ===
if __name__ == "__main__":
    main("P2S8MEICA")  