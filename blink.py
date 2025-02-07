import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from pylsl import StreamInlet, resolve_streams
import time

# Resolve LSL streams and select the EEG stream from Petal
streams = resolve_streams()
eeg_stream = next(s for s in streams if s.name() == "PetalStream_eeg")
inlet = StreamInlet(eeg_stream)

# Define a bandpass filter function (1–10 Hz)
def bandpass_filter(data, lowcut=1, highcut=10, fs=256, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# Set up parameters for plotting
fs = 256                          # Sampling rate (Hz)
window_size = 2 * fs              # 2-second data window
data_buffer_af7 = np.zeros(window_size)
data_buffer_af8 = np.zeros(window_size)
time_axis = np.linspace(-2, 0, window_size)  # X-axis from -2 to 0 seconds

# Read initial EEG data (buffer filling)
print("Collecting initial EEG data...")
for i in range(window_size):
    sample, timestamp = inlet.pull_sample()
    data_buffer_af7[i] = sample[1]  # AF7
    data_buffer_af8[i] = sample[2]  # AF8

# Apply bandpass filter separately to AF7 and AF8
filtered_af7 = bandpass_filter(data_buffer_af7)
filtered_af8 = bandpass_filter(data_buffer_af8)

# Detect blinks based on filtered signals separately
peaks_af7, _ = signal.find_peaks(filtered_af7, height=100, distance=50)
peaks_af8, _ = signal.find_peaks(filtered_af8, height=100, distance=50)

# Initialize plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
line_af7, = ax.plot(time_axis, filtered_af7, label="EEG Signal (AF7)", color='blue')
line_af8, = ax.plot(time_axis, filtered_af8, label="EEG Signal (AF8)", color='green')

# Scatter plots for detected blinks
scatter_af7 = ax.scatter(time_axis[peaks_af7], filtered_af7[peaks_af7], color='red', label="Blink (AF7)")
scatter_af8 = ax.scatter(time_axis[peaks_af8], filtered_af8[peaks_af8], color='orange', label="Blink (AF8)")

ax.set_ylim(-200, 200)
ax.set_xlabel("Time (s)")
ax.set_ylabel("EEG Signal")
ax.set_title("Real-time EEG with Blink Detection")

# Move legend outside of the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Position legend outside

plt.tight_layout()
plt.show()

# Continuous update loop
while True:
    # Shift data buffer to the left (scrolling effect)
    data_buffer_af7 = np.roll(data_buffer_af7, -1)
    data_buffer_af8 = np.roll(data_buffer_af8, -1)

    # Read a new EEG sample
    sample, timestamp = inlet.pull_sample()
    data_buffer_af7[-1] = sample[1]  # AF7
    data_buffer_af8[-1] = sample[2]  # AF8

    # Apply bandpass filter separately to AF7 and AF8
    filtered_af7 = bandpass_filter(data_buffer_af7)
    filtered_af8 = bandpass_filter(data_buffer_af8)

    # Update the plotted waveforms
    line_af7.set_ydata(filtered_af7)
    line_af8.set_ydata(filtered_af8)

    # Detect new peaks for blink detection
    peaks_af7, _ = signal.find_peaks(filtered_af7, height=100, distance=50)
    peaks_af8, _ = signal.find_peaks(filtered_af8, height=100, distance=50)

    # Update scatter plots
    scatter_af7.set_offsets(np.column_stack((time_axis[peaks_af7], filtered_af7[peaks_af7])))
    scatter_af8.set_offsets(np.column_stack((time_axis[peaks_af8], filtered_af8[peaks_af8])))

    plt.pause(1.0 / fs)
