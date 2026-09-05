import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. ACQUISITION PARAMETERS
# ============================================================

fs = 1000              # Sampling frequency [Hz]
trial_duration = 0.5   # Trial duration [s]
n_trials = 300         # Number of trials

samples_per_trial = int(fs * trial_duration)

# Time vector for one trial
t = np.arange(samples_per_trial) / fs
time_ms = t * 1000


# ============================================================
# 2. STIMULUS PARAMETERS
# ============================================================

stimulus_onset_ms = 50      # Stimulus starts 50 ms after trial begins
stimulus_duration_ms = 10   # Visual stimulus duration

stimulus_onset_sample = int(
    stimulus_onset_ms * fs / 1000
)

stimulus_duration_samples = int(
    stimulus_duration_ms * fs / 1000
)


# ============================================================
# 3. CLEAN ERP SIGNAL
# ============================================================

A = 100
alpha = 12
f = 10  # Hz

# ERP is generated relative to stimulus onset
erp_time = t - (stimulus_onset_ms / 1000)

# Before the stimulus, ERP is zero
clean_erp = np.zeros_like(t)

# Only generate ERP after stimulus onset
post_stimulus = erp_time >= 0

clean_erp[post_stimulus] = (
    A
    * erp_time[post_stimulus]
    * np.exp(-alpha * erp_time[post_stimulus])
    * np.sin(2 * np.pi * f * erp_time[post_stimulus])
)


# ============================================================
# 4. GENERATE 300 NOISY EEG TRIALS
# ============================================================

noise_rms = 5

trials = np.zeros(
    (n_trials, samples_per_trial)
)

for i in range(n_trials):

    noise = np.random.normal(
        loc=0,
        scale=noise_rms,
        size=samples_per_trial
    )

    trials[i, :] = clean_erp + noise


# ============================================================
# 5. TRIAL INFORMATION
# ============================================================

trial_start_samples = (
    np.arange(n_trials) * samples_per_trial
)

trial_end_samples = (
    trial_start_samples + samples_per_trial - 1
)

stimulus_samples = (
    trial_start_samples + stimulus_onset_sample
)


# ============================================================
# 6. SAVE DATASET AS NPZ
# ============================================================

np.savez(
    "simulated_eeg_300_trials.npz",

    trials=trials,
    clean_erp=clean_erp,

    time_ms=time_ms,

    fs=fs,
    n_trials=n_trials,
    samples_per_trial=samples_per_trial,

    trial_start_samples=trial_start_samples,
    trial_end_samples=trial_end_samples,

    stimulus_onset_ms=stimulus_onset_ms,
    stimulus_duration_ms=stimulus_duration_ms,
    stimulus_samples=stimulus_samples
)


# ============================================================
# 7. DATASET INFORMATION
# ============================================================

print("NPZ dataset generated successfully")
print("----------------------------------")
print("Number of trials:", n_trials)
print("Samples per trial:", samples_per_trial)
print("Sampling frequency:", fs, "Hz")
print("Trial duration:", trial_duration * 1000, "ms")
print("Stimulus onset:", stimulus_onset_ms, "ms")
print("Stimulus duration:", stimulus_duration_ms, "ms")
print("Trial matrix shape:", trials.shape)


# ============================================================
# 8. VISUALIZE FIRST 3 TRIALS
# ============================================================

n_show = 3

first_trials = trials[:n_show].flatten()

display_time_ms = (
    np.arange(len(first_trials)) / fs * 1000
)

plt.figure(figsize=(14, 5))

plt.plot(
    display_time_ms,
    first_trials,
    linewidth=0.8,
    label="Noisy EEG"
)


# ============================================================
# 9. TRIAL BOUNDARIES AND STIMULUS WINDOWS
# ============================================================

for i in range(n_show):

    trial_start_ms = (
        i * trial_duration * 1000
    )

    trial_end_ms = (
        (i + 1) * trial_duration * 1000
    )

    stimulus_start = (
        trial_start_ms + stimulus_onset_ms
    )

    stimulus_end = (
        stimulus_start + stimulus_duration_ms
    )

    # Trial boundary
    plt.axvline(
        trial_start_ms,
        linestyle="--",
        linewidth=1
    )

    # Stimulus presentation window
    plt.axvspan(
        stimulus_start,
        stimulus_end,
        alpha=0.25
    )

    # Trial label
    plt.text(
        trial_start_ms + 180,
        plt.ylim()[1] * 0.85,
        f"Trial {i + 1}",
        ha="center"
    )

    # Stimulus label
    plt.text(
        stimulus_start,
        plt.ylim()[0] * 0.85,
        "Stimulus",
        rotation=90,
        va="bottom"
    )


# Final boundary
plt.axvline(
    n_show * trial_duration * 1000,
    linestyle="--",
    linewidth=1
)


# ============================================================
# 10. PLOT FORMAT
# ============================================================

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("Simulated EEG Trials with Stimulus Presentation")

plt.grid(True)
plt.tight_layout()

plt.show()