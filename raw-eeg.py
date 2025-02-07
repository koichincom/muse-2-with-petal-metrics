import time
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_streams

streams = resolve_streams()
eeg_stream = None
for stream in streams:
    if stream.type() == 'EEG':
        eeg_stream = stream
        break
if eeg_stream is None:
    raise RuntimeError("No EEG stream found.")
inlet = StreamInlet(eeg_stream)
sample, timestamp = inlet.pull_sample()
num_channels = len(sample)
window_duration = 15
sample_rate = 1000
max_samples = int(window_duration * sample_rate)
data = np.zeros((max_samples, num_channels))
time_axis = np.linspace(-window_duration, 0, max_samples)
plt.ion()
fig, axs = plt.subplots(num_channels, 1, figsize=(10, 8))
if num_channels == 1:
    axs = [axs]
lines = []
for i in range(num_channels):
    line, = axs[i].plot(time_axis, data[:, i])
    axs[i].set_ylabel("Channel " + str(i))
    axs[i].set_ylim(-500, 500)
    lines.append(line)
axs[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
while True:
    sample, timestamp = inlet.pull_sample()
    sample = np.array(sample)
    data = np.roll(data, -1, axis=0)
    data[-1, :] = sample
    for i in range(num_channels):
        lines[i].set_ydata(data[:, i])
    plt.pause(1.0 / sample_rate)