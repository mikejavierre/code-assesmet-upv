# code-assesmet-upv
Code Assesment for PHD Position in UPV:

 
# ECG Signal Preprocessing Pipeline

This repository provides a modular Python pipeline for preprocessing ECG signals, including artifact removal, filtering, and channel quality checks. The goal is to clean raw ECG recordings for offfline analysis.


##  Overview of Pipeline Steps



##  1. Removal of Large Amplitude Spikes

**Function:** `remove_spikes_adaptive_power`

Spikes are detected by comparing local signal energy to an adaptive threshold defined by the local mean and standard deviation in a sliding window. Detected spikes are replaced with the median-filtered value.

- **Window size:** 1000 samples  
- **Spike threshold:** 10 × local std  



## 2. Low-pass Filtering

**Function:** `low_pass_filter`

A 4th-order Butterworth low-pass filter is applied to remove high-frequency noise.

- **Cutoff Frequency:** 50 Hz (suitable for standard ECG signals), for HRV porpouse elevate to at least 100
- **Sampling Frequency (`fs`):** Default 1000 Hz  



## 3. Powerline Interference Removal

**Function:** `notch_filter`

Uses multiple IIR notch filters to remove the fundamental powerline frequency and its harmonics.

- **Base Frequency:** 50 Hz 
- **Number of Harmonics:** 5 (filters up to 250 Hz)  
- **Quality Factor (`Q`):** 30  



---

##  4. Baseline Wander and Offset Removal

**Function:** `baseline_wander_removal`

Applies a high-pass Butterworth filter with symmetric padding to remove slow fluctuations due to respiration and movement.

- **Cutoff Frequency:** 0.5 Hz  
- **Filter Order:** 2  
- **Padding:** 1000 samples (to avoid edge effects)  



##  5. Detection of Disconnected ECG Channels

**Function:** `detect_disconnected_channel`

Uses the median absolute derivative as a simple and effective heuristic for flat signals (e.g., disconnected or failed electrodes) and delete them from the pipeline for optimization porpouse.
It is assumed that disconnected channel is mainly compose by 0s. If noise was present further refinement should be done.






## Files

- `ecg_preprocessing_pipeline.py`: Main code with preprocessing and verification logic.
- `main.py`: Test of the preprocessed pipeline.
- `plot_ecg_gui.py`: Generates a gui to check filtered ECG visually.
- `README.md`: This documentation file.
- `run.sh` : excutable to run both the main and the gui, please change the file path acording to your necessity.








