#!/bin/bash



INPUT_FILE="./ecgConditioningExample.mat"
OUTPUT_FILE="./processed_ecg.mat"
PREPROCESS_SCRIPT="main.py"
GUI_SCRIPT="plot_ecg_gui.py"

echo "Ejecutando preprocesamiento ECG..."
python "$PREPROCESS_SCRIPT" --input "$INPUT_FILE" --output "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo "Error durante el preprocesamiento."
    exit 1
fi


echo "Lanzando visualizador interactivo ECG..."
python "$GUI_SCRIPT" --mat_file_path "$OUTPUT_FILE"