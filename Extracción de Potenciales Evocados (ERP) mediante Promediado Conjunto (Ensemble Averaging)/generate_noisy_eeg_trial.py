import numpy as np
import matplotlib.pyplot as plt

# Sampling frequency
fs = 1000  # Hz

# Trial duration
T = 0.5  # seconds

# Time vector
t = np.arange(0, T, 1 / fs)

# Clean ERP parameters
A = 100
alpha = 12
f = 10  # Hz

# Generate clean ERP signal
s = A * t * np.exp(-alpha * t) * np.sin(2 * np.pi * f * t)

# Noise RMS
noise_rms = 5

# Generate Gaussian white noise
noise = np.random.normal(
    loc=0,
    scale=noise_rms,
    size=len(t)
)

# Noisy EEG trial
x1 = s + noise

# Plot only the contaminated trial
plt.figure(figsize=(10, 4))

plt.plot(t * 1000, x1)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("Single Noisy EEG Trial")
plt.grid(True)

plt.show()