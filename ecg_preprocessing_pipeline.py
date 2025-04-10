import numpy as np
from scipy import signal
from scipy.signal import iirnotch, butter, filtfilt, firwin,medfilt
import logging


def remove_spikes_adaptive_power(ecg, window_size=1000, std_thresh=10):
    """
    Remove spikes in ECG signal using adaptive power-based detection.

    Parameters:
        ecg (np.ndarray): ECG signal, shape (channels, samples)
        window_size (int): Size of the moving window for power estimation
        std_thresh (float): Threshold in terms of local std deviation for spike detection

    Returns:
        np.ndarray: ECG signal with spikes removed
    """
    cleaned_ecg = ecg.copy()
    num_spikes_total = 0

    for ch in range(cleaned_ecg.shape[0]):
        signal = cleaned_ecg[ch]
        energy = signal ** 2

        # Compute local mean and std using moving average
        kernel = np.ones(window_size) / window_size
        local_mean = np.convolve(energy, kernel, mode='same')
        local_std = np.sqrt(np.convolve((energy - local_mean)**2, kernel, mode='same'))

        # Adaptive threshold
        threshold = local_mean + std_thresh * local_std

        # Detect spikes
        spikes = np.where(energy > threshold)[0]
        num_spikes_total += len(spikes)

        # Replace spikes with local median value
        signal[spikes] = medfilt(signal, kernel_size=9)[spikes]

        cleaned_ecg[ch] = signal

    print(f"Removed {num_spikes_total} spikes adaptively.")
    return cleaned_ecg


def low_pass_filter(ecg, fs=1000, cutoff=50):
    """
    Apply a low-pass filter to remove high-frequency noise.
    Parameters:
        ecg (np.ndarray): ECG array (channels x samples)
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency in Hz
    Returns:
        np.ndarray: ECG signal with high-frequency noise removed
    """
    nyq = 0.5 * fs
    b, a = signal.butter(4, cutoff / nyq, btype='low')
    return signal.filtfilt(b, a, ecg, axis=1)




def notch_filter(ecg, fs=1000, base_freq=50, n_harmonics=5, q=30):
    """
    Apply multiple notch filters to remove powerline harmonics (e.g., 50, 100, 150 Hz).
    Parameters:
        ecg (np.ndarray): ECG array (channels x samples)
        fs (int): Sampling frequency
        base_freq (float): Base powerline frequency (e.g., 50 or 60 Hz)
        n_harmonics (int): Number of harmonics to filter
        q (float): Quality factor for the notch filter
    Returns:
        np.ndarray: ECG signal without powerline noise
    """
    filtered = ecg.copy()
   

    for i in range(1, n_harmonics + 1):
        freq = base_freq * i
        b, a = iirnotch(freq, q, fs)
        filtered= filtfilt(b, a, filtered, axis=1)
    return filtered

def baseline_wander_removal(ecg, fs=1000, cutoff=0.5, pad_len=1000):
    """
    Remove baseline wander using high-pass Butterworth filter with reflection padding.
    Parameters:
        ecg (np.ndarray): ECG array (channels x samples)
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency in Hz
        pad_len (int): Length of padding to avoid edge distortion
    Returns:
        np.ndarray:  ECG signal with baseline wander removed
    """
    nyq = 0.5 * fs
    b, a = signal.butter(2, cutoff / nyq, btype='high')

    # Apply padding manually to avoid edge distortion
    ecg_padded = np.pad(ecg, ((0, 0), (pad_len, pad_len)), mode="symmetric")
    ecg_filtered = signal.filtfilt(b, a, ecg_padded, axis=1)
    
    # Remove the padding
    return ecg_filtered[:, pad_len:-pad_len]

def detect_disconnected_channel(ecg, threshold=1):
    """
    Detect disconnected channels using the median absolute derivative.
    Parameters:
        ecg (np.ndarray): ECG array (channels x samples)
        threshold (float): Threshold for detecting disconnected channels
    Returns:
        np.ndarray: Boolean array indicating disconnected channels
    """
    mad = np.median(np.abs(np.diff(ecg, axis=1)), axis=1)
    return mad < threshold



logger = logging.getLogger(__name__)

def verify_pipeline(ecg_raw, fs):
    """
    Verify the preprocessing pipeline for ECG data.
    Parameters:
        ecg_raw (np.ndarray): ECG array (channels x samples)
        fs (int): Sampling frequency
    Returns:
        np.ndarray: Preprocessed ECG signal
    """

    logger.info("Starting ECG preprocessing pipeline.")

    # Step 1: Detect disconnected channels
    disconnected = detect_disconnected_channel(ecg_raw)

    if disconnected.any():
        disconnected_channels = np.where(disconnected)[0]
        logger.warning("Some channels appear to be disconnected.")
        logger.warning(f"Removing disconnected channels: {disconnected_channels.tolist()}")
        ecg_raw = np.delete(ecg_raw, disconnected_channels, axis=0)
    else:
        logger.info("No disconnected channels detected.")

    # Step 2: Remove spikes using adaptive power
    logger.info("Removing spikes...")
    ecg = remove_spikes_adaptive_power(ecg_raw)

    # Step 3: Apply notch filter
    logger.info("Applying notch filter...")
    ecg = notch_filter(ecg, fs)

    # Step 4: Apply low-pass filter
    logger.info("Applying low-pass filter...")
    ecg = low_pass_filter(ecg, fs)

    # Step 5: Baseline wander removal
    logger.info("Removing baseline wander...")
    ecg = baseline_wander_removal(ecg, fs)

    # Step 6: Reinsert disconnected channels as zeros
    if disconnected.any():
        ecg = np.insert(ecg, disconnected_channels, 0, axis=0)
        logger.info(f"Reinserted {len(disconnected_channels)} disconnected channels as zeros.")

    logger.info("ECG preprocessing pipeline completed.")

    return ecg

   
    
   

  
