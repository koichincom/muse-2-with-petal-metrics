from pylsl import StreamInlet, resolve_streams
import numpy as np

_inlet = None # Global inlet cache

'''
get the inlet from the stream
'''
def get_inlet():
    global _inlet
    if _inlet is None:
        print("Resolving streams")
        streams = resolve_streams(wait_time=1)
        eeg_stream = None

        for stream in streams:
            if stream.name() == 'PetalStream_eeg':
                eeg_stream = stream
                break

        if eeg_stream is None:
            print("No EEG stream found")
            exit(1)

        print("Inlet created")
        _inlet = StreamInlet(eeg_stream)

    return _inlet

'''
get average of af7 and af8 channels
You can modify the duration_sec and channels as needed, but fs is fixed at 256Hz
'''
def get_af7_af8(duration_sec=10, fs=256):
    inlet = get_inlet()
    n_samples = int(duration_sec * fs)
    data = []
    
    for _ in range(n_samples):
        try:
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if sample is not None:
                af7 = sample[1]
                af8 = sample[2]
                average = (af7 + af8) / 2
                data.append(average)
        except Exception as e:
            print(f"Error pulling sample: {e}")
    
    if not data:
        print("Warning: No LSL data received")
        pass
        
    return np.array(data)