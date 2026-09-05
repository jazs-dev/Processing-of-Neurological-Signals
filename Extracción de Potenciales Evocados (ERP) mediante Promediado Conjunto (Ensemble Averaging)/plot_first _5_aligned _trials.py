import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = np.load("simulated_eeg_300_trials.npz")

trials = data["trials"]
time_ms = data["time_ms"]

stimulus_onset_ms = int(data["stimulus_onset_ms"])
stimulus_duration_ms = int(data["stimulus_duration_ms"])


# ============================================================
# 2. SELECT FIRST 5 TRIALS
# ============================================================

first_5_trials = trials[:5]


# ============================================================
# 3. CREATE FIGURE WITH 5 SEPARATE PANELS
# ============================================================

fig, axes = plt.subplots(
    5,
    1,
    figsize=(12, 10),
    sharex=True
)


# ============================================================
# 4. PLOT EACH TRIAL
# ============================================================

for i in range(5):

    axes[i].plot(
        time_ms,
        first_5_trials[i],
        linewidth=1
    )

    # Stimulus presentation window
    axes[i].axvspan(
        stimulus_onset_ms,
        stimulus_onset_ms + stimulus_duration_ms,
        alpha=0.25
    )

    axes[i].set_ylabel("Amplitude")

    axes[i].set_title(
        f"Trial {i + 1}"
    )

    axes[i].grid(True)


# ============================================================
# 5. AXIS LABELS
# ============================================================

axes[-1].set_xlabel("Time (ms)")

fig.suptitle(
    "First 5 Aligned Noisy EEG Trials",
    fontsize=14
)

plt.tight_layout()

plt.show()