import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = np.load("simulated_eeg_300_trials.npz")

trials = data["trials"]
clean_erp = data["clean_erp"]
time_ms = data["time_ms"]

stimulus_onset_ms = int(data["stimulus_onset_ms"])
stimulus_duration_ms = int(data["stimulus_duration_ms"])


# ============================================================
# 2. SELECT FIRST 16 TRIALS
# ============================================================

N = 16

selected_trials = trials[:N]


# ============================================================
# 3. ENSEMBLE AVERAGING
# ============================================================

average_16 = np.mean(
    selected_trials,
    axis=0
)


# ============================================================
# 4. SELECT ONE RAW NOISY EEG TRIAL
# ============================================================

raw_trial = trials[0]


# ============================================================
# 5. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True
)


# ============================================================
# 6. PANEL 1 - CLEAN ERP
# ============================================================

axes[0].plot(
    time_ms,
    clean_erp,
    linewidth=2
)

axes[0].axvspan(
    stimulus_onset_ms,
    stimulus_onset_ms + stimulus_duration_ms,
    alpha=0.2
)

axes[0].set_title(
    "Clean ERP Signal - Ground Truth"
)

axes[0].set_ylabel("Amplitude")
axes[0].grid(True)


# ============================================================
# 7. PANEL 2 - RECOVERED ERP
# ============================================================

axes[1].plot(
    time_ms,
    average_16,
    linewidth=1.5
)

axes[1].axvspan(
    stimulus_onset_ms,
    stimulus_onset_ms + stimulus_duration_ms,
    alpha=0.2
)

axes[1].set_title(
    "Recovered ERP - Ensemble Average of 16 Trials"
)

axes[1].set_ylabel("Amplitude")
axes[1].grid(True)


# ============================================================
# 8. PANEL 3 - RAW EEG VS RECOVERED ERP
# ============================================================

axes[2].plot(
    time_ms,
    raw_trial,
    linewidth=0.8,
    label="Raw Noisy EEG Trial"
)

axes[2].plot(
    time_ms,
    average_16,
    linewidth=2,
    label="Recovered ERP (N = 16)"
)

axes[2].axvspan(
    stimulus_onset_ms,
    stimulus_onset_ms + stimulus_duration_ms,
    alpha=0.2
)

axes[2].set_title(
    "Raw Simulated EEG Trial with Recovered ERP"
)

axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Amplitude")

axes[2].legend()
axes[2].grid(True)


# ============================================================
# 9. FIGURE FORMAT
# ============================================================

fig.suptitle(
    "ERP Recovery Using Ensemble Averaging - N = 16",
    fontsize=14
)

plt.tight_layout()

plt.show()