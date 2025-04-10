from ecg_preprocessing_pipeline import * 
import numpy as np
import h5py
import time
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ecg_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="ECG Preprocessing Pipeline")
    parser.add_argument('--input', type=str, default='./ecgConditioningExample.mat',
                        help='Path to the input .mat file with raw ECG')
    parser.add_argument('--output', type=str, default='processed_ecg.mat',
                        help='Path to save the processed ECG .mat file')
    args = parser.parse_args()

    mat_file_path = args.input
    output_file_path = args.output

    try:
        logger.info(f"Loading file: {mat_file_path}")
        with h5py.File(mat_file_path, 'r') as f:
            if "ecg" not in f:
                logger.error("The file does not contain the 'ecg' dataset.")
                raise KeyError("The file does not contain the 'ecg' dataset.")
            if "fs" not in f:
                logger.error("The file does not contain the 'fs' dataset.")
                raise KeyError("The file does not contain the 'fs' dataset.")

            ecg_raw = np.array(f["ecg"]) 
            fs = int(np.array(f["fs"]))

        logger.info(f"Sampling frequency: {fs} Hz")
        logger.info(f"Original shape: {ecg_raw.shape}")

        if ecg_raw.dtype != np.float64:
            logger.error("Input data must be float64")
            raise ValueError("Input data must be float64")

        logger.info("Running preprocessing pipeline...")
        t0 = time.time()
        ecg = verify_pipeline(ecg_raw, fs)
        elapsed = time.time() - t0
        logger.info(f"Preprocessing completed in {elapsed:.4f} seconds.")

        if ecg.dtype != np.float64:
            logger.error("Output data must be float64")
            raise ValueError("Output data must be float64")

        if np.isnan(ecg).any():
            logger.error("Output contains NaN values")
            raise ValueError("Output contains NaN values")

        logger.info("Output validation passed.")

        with h5py.File(output_file_path, 'w') as f:
            f.create_dataset("ecg", data=ecg)
            f.create_dataset("fs", data=fs)
        logger.info(f"Processed ECG data saved to {output_file_path}")

    except Exception as e:
        logger.exception("An unhandled  error occurred during ECG preprocessing:")
        raise

if __name__ == "__main__":
    main()
