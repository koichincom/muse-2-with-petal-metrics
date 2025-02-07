import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from pylsl import StreamInlet, resolve_streams

# Set up the LSL EEG stream
print("Looking for an EEG stream...")
streams = resolve_streams(wait_time=1.0)
# Filter by stream name 'PetalStream_eeg'
eeg_streams = [s for s in streams if s.name() == 'PetalStream_eeg']
if not eeg_streams:
    print("No EEG stream found!")
    exit()

inlet = StreamInlet(eeg_streams[0])
print("Successfully connected to EEG stream")

# Parameters for plotting
fs = 256  # Sampling rate (Muse 2 typically uses 256 Hz)
buffer_length = fs  # 1 second buffer
raw_buffer = deque([0] * buffer_length, maxlen=buffer_length)

# Set a threshold for blink detection.
# You might need to adjust this threshold based on your data characteristics.
blink_threshold = 50  # Example threshold in microvolts (adjust as needed)

# Set up real-time plotting for a single EEG channel (e.g., channel 0)
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(range(buffer_length), list(raw_buffer), lw=2, color='blue', label='EEG Channel')
ax.set_ylim(-100, 100)
ax.set_xlabel("Sample Index")
ax.set_ylabel("EEG Amplitude (µV)")
ax.legend()
plt.tight_layout()

while True:
    sample, timestamp = inlet.pull_sample()
    if sample is not None:
        # For blink detection, choose the channel that shows the blink clearly.
        # Here we assume channel 0. Change the index if necessary.
        value = sample[0]
        raw_buffer.append(value)
        
        # Check if the current value exceeds the blink threshold
        if value > blink_threshold:
            blink = True
        else:
            blink = False
        
        # Update the plot with the current buffer data
        line.set_ydata(list(raw_buffer))
        # Change the line color if a blink is detected
        if blink:
            line.set_color('red')
        else:
            line.set_color('blue')
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)
