
from ecg_preprocessing_pipeline import * 
import numpy as np
import h5py
import time





def main():

   # Ruta al archivo
    mat_file_path = './ecgConditioningExample.mat'

    with h5py.File(mat_file_path, 'r') as f:
        print("Keys:", list(f.keys()))  # e.g. ['ecg', 'fs']
        
        # Obtener los datos
        ecg_raw = np.array(f["ecg"]) 
        fs = int(np.array(f["fs"])[0][0])  
    print("Frecuencia de muestreo:", fs)

    # Opcional: convertir a DataFrame

    print("Original shape:", ecg_raw.shape)
    assert ecg_raw.dtype == np.float64, "Input data must be double (float64)"

    # Preprocessing
    t0 = time.time()
    ecg= verify_pipeline(ecg_raw)

    print("Processing completed in %.4f seconds." % (time.time() - t0))
    assert ecg.dtype == np.float64, "Output data must be float64"
    assert not np.isnan(ecg).any(), "Output contains NaN values"
    print("All checks passed.")





if __name__ == "__main__":
    main()