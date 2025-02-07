import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from pylsl import StreamInlet, resolve_streams

# Parameters for EEG and blink detection
fs = 256                        # Sampling rate for Muse 2 (Hz)
frame_interval = 10            # seconds per frame
frame_size = int(fs * frame_interval)  # Number of samples per frame

channel_index = 1               # Choose the channel for blink detection (e.g., AF7)

# Advanced detection parameters
zscore_threshold = 3.0          # A blink is detected if the z-score exceeds this value
refractory_period = 0.3         # In seconds, ignore further detections for 300 ms

# -------------------------------
# Set up the LSL EEG stream
print("Looking for an EEG stream")
streams = resolve_streams(wait_time=1.0)
# Filter by stream name 'PetalStream_eeg'
eeg_streams = [s for s in streams if s.name() == 'PetalStream_eeg']
if not eeg_streams:
    print("No EEG stream found!")
    exit()

inlet = StreamInlet(eeg_streams[0])
print("Successfully connected to EEG stream")

# -------------------------------
# Set up buffers for the sliding window (for blink detection) and time stamping
raw_buffer = deque(maxlen=frame_size)
time_buffer = deque(maxlen=frame_size)

# -------------------------------
# Function to compute z-score
def compute_zscore(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    # Avoid division by zero
    if std_val == 0:
        return np.zeros_like(data)
    return (data - mean_val) / std_val

# -------------------------------
# Set up Matplotlib for real-time plotting of raw EEG with blink markers
plt.ion()
fig, ax = plt.subplots(figsize=(12, 4))
line, = ax.plot([], [], lw=1.5, color='blue', label=f'EEG (Channel {channel_index})')
ax.set_xlabel("Time (s)")
ax.set_ylabel("EEG Amplitude (µV)")
ax.set_title("Real-Time EEG Signal with Blink Detection")
ax.set_ylim(-1500, 1500)
ax.legend()
plt.tight_layout()

start_time = time.time()
last_blink_time = 0

# List to keep track of blink markers for later removal if needed
blink_marker_lines = []

# -------------------------------
# Main loop: acquire data, compute blink metric, and update the plot
while True:
    # Pull a sample from the LSL stream (non-blocking)
    sample, ts = inlet.pull_sample(timeout=0.0)
    if sample is not None:
        current_time = time.time() - start_time
        # Append current sample (from chosen channel) and timestamp to buffers
        raw_buffer.append(sample[channel_index])
        time_buffer.append(current_time)
        
        # When we've collected enough samples for one frame, process the data
        if len(raw_buffer) == frame_size:
            data_array = np.array(raw_buffer)
            # Compute the derivative of the signal in the frame
            derivative = np.diff(data_array)
            # Compute the absolute derivative values
            abs_deriv = np.abs(derivative)
            # Compute z-score for the derivative values
            z_scores = compute_zscore(abs_deriv)
            max_z = np.max(z_scores)
            
            # Check if blink is detected: if max z-score exceeds the threshold and refractory period has passed
            if max_z > zscore_threshold and (current_time - last_blink_time) > refractory_period:
                last_blink_time = current_time
                print(f"Blink detected at {current_time:.2f}s with z-score {max_z:.2f}")
                # Mark the blink on the plot with a vertical red dashed line
                marker_line = ax.axvline(x=current_time, color='red', linestyle='--', linewidth=1)
                blink_marker_lines.append(marker_line)
            
            # Optionally, clear the raw buffer to start a new frame.
            # (Alternatively, you can use overlapping windows for continuous detection.)
            raw_buffer.clear()
            time_buffer.clear()
        
        # Update the plot with current raw EEG data
        line.set_data(np.array(time_buffer), np.array(raw_buffer))
        # Adjust x-axis limits so the plot scrolls; show the last frame_interval seconds
        if time_buffer:
            ax.set_xlim(max(0, current_time - frame_interval), current_time + 0.1)
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    time.sleep(1.0 / fs)
