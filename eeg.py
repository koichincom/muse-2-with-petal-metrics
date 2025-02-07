import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from pylsl import StreamInlet, resolve_streams
from scipy.signal import welch

# Define EEG frequency bands (in Hz)
EEG_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 100)
}

# Sampling rate for Muse 2 (Hz)
fs = 256
# Frame interval (seconds) and number of samples per frame
frame_interval = 0.5  # 0.5 seconds per frame
frame_size = int(fs * frame_interval)

# Set up LSL EEG stream
print("Looking for an EEG stream")
streams = resolve_streams(wait_time=1.0)
# Filter by stream name 'PetalStream_eeg'
eeg_streams = [s for s in streams if s.name() == 'PetalStream_eeg']
if not eeg_streams:
    print("No EEG stream found!")
    exit()

inlet = StreamInlet(eeg_streams[0])
print("Successfully connected to EEG stream")

# Set up buffers:
# raw_buffer will collect frame_size samples from one EEG channel (e.g., channel 0)
raw_buffer = deque(maxlen=frame_size)
# For each band, keep a time series (store the last 100 computed frames)
num_frames_to_display = 100
band_time_series = {band: deque([0]*num_frames_to_display, maxlen=num_frames_to_display)
                    for band in EEG_BANDS}

# Function to compute band power using Welch's method
def compute_band_power(data, lowcut, highcut, fs):
    # Compute power spectral density over the entire frame
    freqs, psd = welch(data, fs, nperseg=len(data))
    # Find indices corresponding to the desired frequency band
    idx_band = np.logical_and(freqs >= lowcut, freqs <= highcut)
    # Return the mean power in the band
    return np.mean(psd[idx_band])

# Set up Matplotlib for real-time plotting
plt.ion()
fig, axes = plt.subplots(len(EEG_BANDS), 1, figsize=(10, 8), sharex=True)
lines = {}
x_axis = np.arange(num_frames_to_display)  # x-axis for frames
band_names = list(EEG_BANDS.keys())
for i, band in enumerate(band_names):
    # Plot initial zero values for the time series
    line, = axes[i].plot(x_axis, list(band_time_series[band]), label=band)
    axes[i].set_ylabel(f"{band} Power")
    axes[i].legend(loc="upper right")
    axes[i].set_ylim(0, 1)
    lines[band] = line
axes[-1].set_xlabel("Frame Index")
plt.tight_layout()

# Main loop: Collect samples, compute band power per frame, and update the plots
while True:
    # Pull a new sample from the inlet (sample is a list of channel values)
    sample, timestamp = inlet.pull_sample()
    if sample is not None:
        # Append channel 0's data to the raw buffer (you can choose another channel if desired)
        raw_buffer.append(sample[0])
        
        # When we've collected enough samples for one frame:
        if len(raw_buffer) == frame_size:
            data_array = np.array(raw_buffer)
            # For each EEG band, compute its power and update the time series
            for band, (low, high) in EEG_BANDS.items():
                power = compute_band_power(data_array, low, high, fs)
                # Optionally, apply scaling (e.g., logarithmic scaling)
                scaled_power = np.log1p(power)  # Use log(1 + power) to expand small values
                band_time_series[band].append(scaled_power)
            
            # Clear the raw buffer to start a new frame.
            # (Alternatively, you can implement an overlapping window by sliding the window 
            # instead of clearing it.)
            raw_buffer.clear()
            
            # Update each band's plot with the new time series
            for band in band_names:
                new_data = list(band_time_series[band])
                lines[band].set_ydata(new_data)
                # Adjust the y-axis limit dynamically (if needed)
                current_max = max(new_data)
                axes[band_names.index(band)].set_ylim(0, current_max * 1.2 if current_max > 0 else 1)
            
            # Redraw the figure
            fig.canvas.draw()
            fig.canvas.flush_events()
    
    # Short sleep to prevent a busy loop
    time.sleep(0.01)
