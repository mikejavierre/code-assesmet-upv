# code-assesmet-upv
Code Assesment for PHD Position in UPV:

 
ECG Signal Preprocessing Pipeline

This repository provides a complete ECG signal preprocessing pipeline implemented in Python. It includes the following standard preprocessing steps:

1. **Removal of large amplitude spikes (artifacts)**
2. **Low-pass filtering**
3. **Powerline interference removal (notch filter)**
4. **Baseline wander and offset removal**
5. **Detection of disconnected ECG channels**

The implementation is modular, allowing individual functions to be reused or modified independently.

---

## 📁 Files

- `ecg_preprocessing_pipeline.py`: Main code with preprocessing and verification logic.
- `ecgConditioningExample.mat`: Example file (not included here, assumed to be provided).
- `README.md`: This documentation file.

---

## ⚙️ Dependencies

- `numpy`
- `scipy`
- `matplotlib` (for optional visualization)

Install them with:

