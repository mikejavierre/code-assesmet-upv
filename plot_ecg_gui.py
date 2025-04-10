import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import argparse

def load_ecg_data(mat_file_path):
    with h5py.File(mat_file_path, 'r') as f:
        ecg = np.array(f["ecg"])
        fs = int(np.array(f["fs"]))  
    return ecg, fs 

class ECGViewer:
    def __init__(self, master, ecg_data, fs, window_sec=5):
        self.master = master
        self.ecg = ecg_data 
        self.fs = fs
        self.num_channels = ecg_data.shape[0]
        self.window_size = int(fs * window_sec)
        self.start = 0
        self.end = self.window_size
        self.current_channel = 0

        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Container for canvas
        container = tk.Frame(self.master)
        container.grid(row=0, column=0, sticky="nsew")

        # Subplots
        self.fig, (self.ax, self.ax_fft) = plt.subplots(
            2, 1, figsize=(10, 4.5),
            gridspec_kw={'height_ratios': [2, 1]}
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Botonera fija debajo del canvas
        self.button_frame = tk.Frame(self.master)
        self.button_frame.grid(row=1, column=0, sticky="ew", pady=5)

        self.prev_button = tk.Button(self.button_frame, text="⏪ Previous", command=self.prev_window)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.button_frame, text="Next ⏩", command=self.next_window)
        self.next_button.pack(side=tk.LEFT)

        self.zoom_in_button = tk.Button(self.button_frame, text="+ Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT)

        self.zoom_out_button = tk.Button(self.button_frame, text="- Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT)

        self.channel_var = tk.StringVar(value="0")
        self.channel_menu = tk.OptionMenu(
            self.button_frame,
            self.channel_var,
            *[str(i) for i in range(self.num_channels)],
            command=self.change_channel
        )
        self.channel_menu.pack(side=tk.LEFT)
        tk.Label(self.button_frame, text="Channel").pack(side=tk.LEFT)

        self.update_plot()


    def update_plot(self):
        self.ax.clear()
        self.ax_fft.clear()

        # Señal en dominio temporal
        t = np.arange(self.start, self.end) / self.fs
        signal = self.ecg[self.current_channel, self.start:self.end]
        self.ax.plot(t, signal)
        self.ax.set_title(f"ECG Signal | Channel {self.current_channel} | Samples {self.start}-{self.end}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

        # Espectro centrado con fftshift
        N = len(signal)
        fft_vals = np.fft.fft(signal)
        fft_vals_shifted = np.fft.fftshift(np.abs(fft_vals) / N)
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/self.fs))
        self.ax_fft.plot(freqs, fft_vals_shifted)
        self.ax_fft.set_xlim(-self.fs/2, self.fs/2)
        self.ax_fft.set_title("Frequency Spectrum (Centered FFT)")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Magnitude")
        self.ax_fft.grid(True)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def next_window(self):
        if self.end + self.window_size <= self.ecg.shape[1]:
            self.start += self.window_size
            self.end += self.window_size
            self.update_plot()

    def prev_window(self):
        if self.start - self.window_size >= 0:
            self.start -= self.window_size
            self.end -= self.window_size
            self.update_plot()

    def zoom_in(self):
        if self.window_size > int(self.fs * 1):
            self.window_size = int(self.window_size * 0.5)
            self.end = self.start + self.window_size
            self.update_plot()

    def zoom_out(self):
        if self.end + self.window_size <= self.ecg.shape[1]:
            self.window_size = int(self.window_size * 2)
            self.end = self.start + self.window_size
            self.update_plot()

    def change_channel(self, value):
        self.current_channel = int(value)
        self.update_plot()

def main():
    parser = argparse.ArgumentParser(description="ECG Signal Viewer with Channel Selector and FFT")
    parser.add_argument('--mat_file_path', type=str, default='./processed_ecg.mat',
                        help='Path to the .mat file containing ECG and fs')
    args = parser.parse_args()

    mat_file_path = args.mat_file_path
    ecg_data, fs = load_ecg_data(mat_file_path)

    print("Frecuencia de muestreo:", fs)
    print("Tamaño de la señal:", ecg_data.shape)

    root = tk.Tk()
    root.title("ECG Signal Viewer with Channel Selector and FFT")
    app = ECGViewer(root, ecg_data, fs, window_sec=5)
    root.mainloop()

if __name__ == "__main__":
    main()
