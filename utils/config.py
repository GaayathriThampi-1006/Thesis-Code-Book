# utils/config.py
# This is the configuration file for VajraCode. 

# It contains the user paths, output path, data filepaths and various parameters for Pilot 1 and 2 in the Chakra fMRI project. 

# USAGE
# Data Organization:
# Your local folder structure must exactly mirror the CHAKRA_PREP structure found in the SurfDrive folder.
# Set the appropriate USER_PATH by uncommenting your path and commenting out all unused paths.

##############################################
###               USER PATHs               ###
##############################################

#JANICK_PATH = "C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/"
#AIDAN_PATH = "/Users/aidanlyon/Documents/"
GAAYATHRI_PATH = "C:/Users/gaaya/OneDrive/Desktop/UVA Thesis stuff/CHAKRA_PREP/"
#PARSA_PATH = ""
#OLIVIER_PATH =""

#USER_PATH = AIDAN_PATH
#USER_PATH = JANICK_PATH
USER_PATH = GAAYATHRI_PATH
#USER_PATH = PARSA_PATH
#USER_PATH = OLIVIER_PATH

##############################################
### Global parameters for all of VajraCode ###
##############################################

#OUTPUT_PATH = "/Users/aidanlyon/Documents/GitHub/VajraCode/output" # Output path for VajraCode
OUTPUT_PATH = USER_PATH + "GitHub/ChakrafMRI/output" # Output path for VajraCode

FD_THRESHOLD = 0.5  # Set the FD threshold (e.g., 0.2 or 0.5)

##############################################
### Parameters specific to PILOT 1         ###      # after adding the fieldmaps and rerunning fMRIprep, edit file paths (P1_PATH, P1_FUNC... etc.) in this script. 
##############################################

P1_TR = 2.2 # TR in seconds 


# file paths

P1_PATH = USER_PATH + "GitHub/PILOT1/sub-01/"   # deleted "/ses-pre" after updating the data on SurfDrive (March 21, Janick)

# Masks 
P1_BRAIN_MASK_PATHS_NATIVE = { 
    'P1S2': 'func/sub-01_task-S2_space-T1w_desc-brain_mask.nii.gz',     
    'P1S3': 'func/sub-01_task-S3_space-T1w_desc-brain_mask.nii.gz',   
    'P1S4': 'func/sub-01_task-S4_space-T1w_desc-brain_mask.nii.gz', 
    'P1S5': 'func/sub-01_task-S5_space-T1w_desc-brain_mask.nii.gz'
}

P1_BRAIN_MASK_PATHS_NATIVE_REFINED = {
    'P1S2': 'func/sub-01_task-S2_space-T1w_desc-brain_mask_refined.nii.gz',     
    'P1S3': 'func/sub-01_task-S3_space-T1w_desc-brain_mask_refined.nii.gz',   
    'P1S4': 'func/sub-01_task-S4_space-T1w_desc-brain_mask_refined.nii.gz', 
    'P1S5': 'func/sub-01_task-S5_space-T1w_desc-brain_mask_refined.nii.gz' # refined with the generate_refined_mask function
}

P1_GM_MASK_PATHS_NATIVE = {
    'P1S2': 'func/P1S2_NATIVE_GM_mask_thr03.nii.gz',     
    'P1S3': 'func/P1S3_NATIVE_GM_mask_thr03.nii.gz',   
    'P1S4': 'func/P1S4_NATIVE_GM_mask_thr03.nii.gz', 
    'P1S5': 'func/P1S5_NATIVE_GM_mask_thr03.nii.gz' # probabilistic anatomical gray matter mask interpolated to functional mask with a threshold of 0.3
}

# Resampled Native Schaefer Atlases
P1_SCHAEFER_NAT_7YEO = USER_PATH + "GitHub/SCHAEFER_400/Schaefer400_7Yeo_native_P1.nii.gz"
P1_SCHAEFER_NAT_17YEO = USER_PATH + "GitHub/SCHAEFER_400/Schaefer400_17Yeo_native_P1.nii.gz"

P1_ANAT = P1_PATH + "anat/sub-01_desc-preproc_T1w.nii.gz"
P1_GM_MASK_NATIVE = P1_PATH + "anat/sub-01_label-GM_probseg.nii.gz"
P1_GM_MASK_MNI = P1_PATH + "anat/sub-01_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz"   # probabilistic GM mask

P1S5_GM_MASK = P1_PATH + "func/P1S5_GM_mask_thr03.nii.gz"



# Functional Images
P1_FUNC_IMG_PATHS_MNI = { 
    'P1S2': 'func/sub-01_task-S2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',     
    'P1S3': 'func/sub-01_task-S3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',   
    'P1S4': 'func/sub-01_task-S4_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', 
    'P1S5': 'func/sub-01_task-S5_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
}
P1_FUNC_IMG_PATHS_NATIVE = { 
    'P1S2': 'func/sub-01_task-S2_space-T1w_desc-preproc_bold.nii.gz',       
    'P1S3': 'func/sub-01_task-S3_space-T1w_desc-preproc_bold.nii.gz',   
    'P1S4': 'func/sub-01_task-S4_space-T1w_desc-preproc_bold.nii.gz', 
    'P1S5': 'func/sub-01_task-S5_space-T1w_desc-preproc_bold.nii.gz'
}

# confounds
P1_CONFOUNDS = { # Dictionary mapping confounds relative to the main path 
    'P1S2': 'func/sub-01_task-S2_desc-confounds_timeseries.tsv',
    'P1S3': 'func/sub-01_task-S3_desc-confounds_timeseries.tsv', 
    'P1S4': 'func/sub-01_task-S4_desc-confounds_timeseries.tsv',
    'P1S5': 'func/sub-01_task-S5_desc-confounds_timeseries.tsv' 
}

# raw physio data
P1_PHYSIO = { 
    'P1S2': 'C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/GitHub/PILOT1/sub-01/physio files/SCANPHYSLOG20240925134823 (S2).log',
    'P1S3': 'C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/GitHub/PILOT1/sub-01/physio files/SCANPHYSLOG20240925141255 (S3).log', 
    'P1S4': 'C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/GitHub/PILOT1/sub-01/physio files/SCANPHYSLOG20240925143349 (S4).log',
    'P1S5': 'C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/GitHub/PILOT1/sub-01/physio files/SCANPHYSLOG20240925145507 (S5).log' 
}

# number of volumes per scan
P1_VOLNUM = {
    'P1S2': 337,
    'P1S3': 542, 
    'P1S4': 542,
    'P1S5': 542 
}

# Timings of condition in a given scan (silent part of conditions, except for YNS which has audio)
P1_TIMINGS = { 
    "P1S2": { 
        "SRS": {"start": 30, "end": 330}, # start after instructions, end before ending chime
        "YRS": {"start": 435, "end": 742}, 
    }, 
    "P1S3": { 
        "YRS": {"start": 95, "end": 403}, 
        "YNS": {"start": 415, "end": 1110}, 
    }, 
    "P1S4": { 
        "YRS": {"start": 95, "end": 403}, 
        "Ajna": {"start": 504, "end": 1110},
    }, 
    "P1S5": { 
        "YRS": {"start": 95, "end": 403}, 
        "Anahata": {"start": 495, "end": 1103},
    } 
}

# Assign baseline and meditation based on existing values
for session, timings in P1_TIMINGS.items():
    first_condition, second_condition = timings.keys()
    timings["baseline"] = timings[first_condition]
    timings["meditation"] = timings[second_condition]

##############################################
### Parameters specific to PILOT2          ###
##############################################

P2_TR = 2.2 # TR in seconds 

# base file path
P2_PATH = USER_PATH + "GitHub/PILOT2/sub-02/" 

# resampled native schaefer atlases
P2_SCHAEFER_NAT_7YEO = USER_PATH + "GitHub/SCHAEFER_400/Schaefer400_7Yeo_native_P2.nii.gz"
P2_SCHAEFER_NAT_17YEO = USER_PATH + "GitHub/SCHAEFER_400/Schaefer400_17Yeo_native_P2.nii.gz"

P2_ANAT = P2_PATH + "anat/sub-02_desc-preproc_T1w.nii.gz"
P2_GM_MASK_NATIVE = P2_PATH + "anat/sub-02_label-GM_probseg.nii.gz"
P2_GM_MASK_MNI = P2_PATH + "anat/sub-02_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz"   # probabilistic GM mask


P2S7ME_GM_MASK = P2_PATH + "func/P2S7ME_GM_mask_thr03.nii.gz"


P2_GM_MASK_PATHS_NATIVE = { 
    'P2S3SE': 'func/P2S3ME_NATIVE_GM_mask_thr03.nii.gz',     
    'P2S4ME': 'func/P2S4ME_NATIVE_GM_mask_thr03.nii.gz',   
    'P2S5ME': 'func/P2S5ME_NATIVE_GM_mask_thr03.nii.gz', 
    'P2S6SE': 'func/P2S6SE_NATIVE_GM_mask_thr03.nii.gz',
    'P2S7ME': 'func/P2S7ME_NATIVE_GM_mask_thr03.nii.gz',     
    'P2S8ME': 'func/P2S8ME_NATIVE_GM_mask_thr03.nii.gz'   # probabilistic anatomical gray matter mask interpolated to functional mask with a threshold of 0.3
}

P2_BRAIN_MASK_PATHS_NATIVE = { 
    'P2S3SE': 'func/sub-02_task-S3_space-T1w_desc-brain_mask.nii.gz',     
    'P2S4ME': 'func/sub-02_task-S4_space-T1w_desc-brain_mask.nii.gz',   
    'P2S5ME': 'func/sub-02_task-S5_space-T1w_desc-brain_mask.nii.gz', 
    'P2S6SE': 'func/sub-02_task-S6_space-T1w_desc-brain_mask.nii.gz',
    'P2S7ME': 'func/sub-02_task-S7_space-T1w_desc-brain_mask.nii.gz',     
    'P2S8ME': 'func/sub-02_task-S8_space-T1w_desc-brain_mask.nii.gz',

    'P2S4MEICA': 'func/sub-02_task-S4_space-T1w_desc-brain_mask.nii.gz',   # same brain masks for MEICA
    'P2S5MEICA': 'func/sub-02_task-S5_space-T1w_desc-brain_mask.nii.gz', 
    'P2S7MEICA': 'func/sub-02_task-S7_space-T1w_desc-brain_mask.nii.gz',     
    'P2S8MEICA': 'func/sub-02_task-S8_space-T1w_desc-brain_mask.nii.gz'      
}

P2_BRAIN_MASK_PATHS_NATIVE_REFINED = { 
    'P2S3SE': 'func/sub-02_task-S3_space-T1w_desc-brain_mask_refined.nii.gz',     
    'P2S4ME': 'func/sub-02_task-S4_space-T1w_desc-brain_mask_refined.nii.gz',   
    'P2S5ME': 'func/sub-02_task-S5_space-T1w_desc-brain_mask_refined.nii.gz', 
    'P2S6SE': 'func/sub-02_task-S6_space-T1w_desc-brain_mask_refined.nii.gz',
    'P2S7ME': 'func/sub-02_task-S7_space-T1w_desc-brain_mask_refined.nii.gz',     
    'P2S8ME': 'func/sub-02_task-S8_space-T1w_desc-brain_mask_refined.nii.gz'   
}

P2_MEICA_GOODSIGNAL_MASK = { 
    'P2S4MEICA': 'func/sub-02_task-S4_tedana/desc-adaptiveGoodSignal_mask_inT1w.nii.gz',   
    'P2S5MEICA': 'func/sub-02_task-S5_tedana/desc-adaptiveGoodSignal_mask_inT1w.nii.gz', 
    'P2S7MEICA': 'func/sub-02_task-S7_tedana/desc-adaptiveGoodSignal_mask_inT1w.nii.gz',     
    'P2S8MEICA': 'func/sub-02_task-S8_tedana/desc-adaptiveGoodSignal_mask_inT1w.nii.gz'   
}

P2_MEICA_GOODSIGNAL_MASK_THR3 = { 
    'P2S4MEICA': 'func/sub-02_task-S4_tedana/P2S4MEICA_goodsignal_binary_thr3.nii.gz',   # binarized good signal mask including only voxels with good signal in all echoes
    'P2S5MEICA': 'func/sub-02_task-S5_tedana/P2S5MEICA_goodsignal_binary_thr3.nii.gz', 
    'P2S7MEICA': 'func/sub-02_task-S7_tedana/P2S7MEICA_goodsignal_binary_thr3.nii.gz',     
    'P2S8MEICA': 'func/sub-02_task-S8_tedana/P2S8MEICA_goodsignal_binary_thr3.nii.gz'   
}

P2_MEICA_GM_MASK_PATHS_NATIVE = { 
    'P2S4MEICA': 'func/sub-02_task-S4_tedana/P2S4MEICA_NATIVE_GM_mask_thr03.nii.gz',   # probabilistic anatomical gray matter mask interpolated to functional mask (tedana to t1w) with a threshold of 0.3
    'P2S5MEICA': 'func/sub-02_task-S5_tedana/P2S5MEICA_NATIVE_GM_mask_thr03.nii.gz', 
    'P2S7MEICA': 'func/sub-02_task-S7_tedana/P2S7MEICA_NATIVE_GM_mask_thr03.nii.gz',     
    'P2S8MEICA': 'func/sub-02_task-S8_tedana/P2S8MEICA_NATIVE_GM_mask_thr03.nii.gz'  
}

P2_FUNC_IMG_PATHS_NATIVE = { 
    'P2S3SE': 'func/sub-02_task-S3_space-T1w_desc-preproc_bold.nii.gz',     
    'P2S4ME': 'func/sub-02_task-S4_space-T1w_desc-preproc_bold.nii.gz',   # ME scans are optimally combined in an fMRIprep-integrated tedana workflow. No ICA applied
    'P2S5ME': 'func/sub-02_task-S5_space-T1w_desc-preproc_bold.nii.gz', 
    'P2S6SE': 'func/sub-02_task-S6_space-T1w_desc-preproc_bold.nii.gz',
    'P2S7ME': 'func/sub-02_task-S7_space-T1w_desc-preproc_bold.nii.gz',     
    'P2S8ME': 'func/sub-02_task-S8_space-T1w_desc-preproc_bold.nii.gz',  
}

P2_MEICA_FUNC_IMG_PATHS_NATIVE = { # coregistered to anatomical image post tedana, using the coregister_tedana_to-t1w.sh script.
    'P2S4MEICA': 'func/sub-02_task-S4_tedana/desc-denoised_bold_inT1w.nii.gz',   # MEICA scans are optimally combined and ICA-denoised in a post fMRIprep tedana workflow using robust ICA with 30 runs 
    'P2S5MEICA': 'func/sub-02_task-S5_tedana/desc-denoised_bold_inT1w.nii.gz', 
    'P2S7MEICA': 'func/sub-02_task-S7_tedana/desc-denoised_bold_inT1w.nii.gz',     
    'P2S8MEICA': 'func/sub-02_task-S8_tedana/desc-denoised_bold_inT1w.nii.gz',  
}

P2_FUNC_IMG_PATHS_MNI = { 
    'P2S3SE': 'func/sub-02_task-S3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',     
    'P2S4ME': 'func/sub-02_task-S4_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',   # ME scans are optimally combined in an fMRIprep-integrated tedana workflow. No ICA applied
    'P2S5ME': 'func/sub-02_task-S5_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', 
    'P2S6SE': 'func/sub-02_task-S6_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',  
    'P2S7ME': 'func/sub-02_task-S7_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',  
    'P2S8ME': 'func/sub-02_task-S8_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'  
}

P2_PHYSIO = { # 
    'P2S5ME': 'C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/GitHub/PILOT2/sub-02/physio files/SCANPHYSLOG20241204204136 (S5).log',
    'P2S6SE': 'C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/GitHub/PILOT2/sub-02/physio files/SCANPHYSLOG20241204210327 (S6).log', 
    'P2S7ME': 'C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/GitHub/PILOT2/sub-02/physio files/SCANPHYSLOG20241204212901 (S7).log',
    'P2S8ME': 'C:/Users/janic/Desktop/Alles Jan 2025/AnahataChakra/Data/GitHub/PILOT2/sub-02/physio files/SCANPHYSLOG20241204215126 (S8).log' 
}

P2_VOLNUM = { # 
    'P2S4ME': 275,
    'P2S5ME': 433,
    'P2S6SE': 487, 
    'P2S7ME': 487,
    'P2S8ME': 487 
}

P2_TIMINGS = {  # Timings for each session and condition
      
    "P2S3SE": {
        "moviesx": {"start": 10, "end": 300},       # preliminary plotting only: replace with the actual movie watching task timings! 
        "moviesy": {"start": 300, "end": 600},      # preliminary plotting only: replace with the actual movie watching task timings! 
    },
    "P2S4ME": {
        "moviesx": {"start": 10, "end": 300},       # preliminary plotting only: replace with the actual movie watching task timings! 
        "moviesy": {"start": 300, "end": 600},      # preliminary plotting only: replace with the actual movie watching task timings! 
    },
    "P2S4MEICA": {
    "moviesx": {"start": 10, "end": 300},           # preliminary plotting only: replace with the actual movie watching task timings! 
    "moviesy": {"start": 300, "end": 600},          # preliminary plotting only: replace with the actual movie watching task timings! 
    },
    "P2S5ME": { 
        "SRS": {"start": 30, "end": 471}, 
        "YRS": {"start": 548, "end": 952},
    },
    "P2S5MEICA": { 
        "SRS": {"start": 30, "end": 471}, 
        "YRS": {"start": 548, "end": 952},
    },
   "P2S6SE": { 
        "YRS": {"start": 68, "end": 472}, 
        "Manipura": {"start": 540, "end": 1067},
    }, 
    "P2S7ME": { 
        "YRS": {"start": 68, "end": 472}, 
        "Anahata": {"start": 545, "end": 1065}, 
    }, 
    "P2S7MEICA": { # same as P2S7ME
        "YRS": {"start": 68, "end": 472}, 
        "Anahata": {"start": 545, "end": 1065},
    }, 
    "P2S8ME": { 
        "YRS": {"start": 68, "end": 472}, 
        "Ajna": {"start": 543, "end": 1067},
    }, 
    "P2S8MEICA": {
        "YRS": {"start": 68, "end": 472}, 
        "Ajna": {"start": 543, "end": 1067},
    } 
}

# Assign baseline and meditation based on existing values
for session, timings in P2_TIMINGS.items():
    first_condition, second_condition = timings.keys()
    timings["baseline"] = timings[first_condition]
    timings["meditation"] = timings[second_condition]

P2_CONFOUNDS = { # Dictionary mapping confounds relative to the main path 
    'P2S3SE': 'func/sub-02_task-S3_desc-confounds_timeseries.tsv', 
    'P2S4ME': 'func/sub-02_task-S4_desc-confounds_timeseries.tsv',  
    'P2S5ME': 'func/sub-02_task-S5_desc-confounds_timeseries.tsv',
    'P2S6SE': 'func/sub-02_task-S6_desc-confounds_timeseries.tsv', 
    'P2S7ME': 'func/sub-02_task-S7_desc-confounds_timeseries.tsv',
    'P2S8ME': 'func/sub-02_task-S8_desc-confounds_timeseries.tsv',
}

P2_MEICA_CONFOUNDS = { # Same fMRIprep confounds as for ME. Confound regression after tedana ICA should only be applied with good reasoning. These files are here for FD scrubbring. 
    'P2S4MEICA': 'func/sub-02_task-S4_desc-confounds_timeseries.tsv',  
    'P2S5MEICA': 'func/sub-02_task-S5_desc-confounds_timeseries.tsv',
    'P2S7MEICA': 'func/sub-02_task-S7_desc-confounds_timeseries.tsv',
    'P2S8MEICA': 'func/sub-02_task-S8_desc-confounds_timeseries.tsv',
}

P2_MEICA_CONFOUNDS_TEDANA = {  # this is a tedana specific set of rmse confounds, doesn't contain the classic confounds
    'P2S4MEICA': 'func/sub-02_task-S4_tedana/desc-confounds_timeseries.tsv',   
    'P2S5MEICA': 'func/sub-02_task-S5_tedana/desc-confounds_timeseries.tsv', 
    'P2S7MEICA': 'func/sub-02_task-S7_tedana/desc-confounds_timeseries.tsv',     
    'P2S8MEICA': 'func/sub-02_task-S8_tedana/desc-confounds_timeseries.tsv'   
}
print("config.py successfully loaded yay")