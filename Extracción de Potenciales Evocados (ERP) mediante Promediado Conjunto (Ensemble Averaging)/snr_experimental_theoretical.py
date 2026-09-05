import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = np.load("simulated_eeg_300_trials.npz")

trials = data["trials"]
clean_erp = data["clean_erp"]

n_trials = int(data["n_trials"])


# ============================================================
# 2. SIMULATION NOISE LEVEL
# ============================================================

# Same RMS noise level used when generating the dataset
noise_rms = 5

noise_power = noise_rms ** 2


# ============================================================
# 3. CLEAN ERP SIGNAL POWER
# ============================================================

signal_power = np.mean(clean_erp ** 2)


# ============================================================
# 4. NUMBER OF TRIALS
# ============================================================

N_values = np.arange(1, n_trials + 1)


# ============================================================
# 5. ARRAYS TO STORE SNR VALUES
# ============================================================

snr_experimental = np.zeros(n_trials)
snr_theoretical = np.zeros(n_trials)


# ============================================================
# 6. CUMULATIVE ENSEMBLE AVERAGING
# ============================================================

cumulative_sum = np.zeros_like(clean_erp)

for i in range(n_trials):

    # Add current trial
    cumulative_sum += trials[i]

    # Current number of averaged trials
    N = i + 1

    # Ensemble average
    average_signal = cumulative_sum / N


    # ========================================================
    # 7. EXPERIMENTAL RESIDUAL NOISE
    # ========================================================

    residual_noise = average_signal - clean_erp

    residual_noise_power = np.mean(
        residual_noise ** 2
    )


    # ========================================================
    # 8. EXPERIMENTAL SNR
    # ========================================================

    snr_experimental[i] = 10 * np.log10(
        signal_power / residual_noise_power
    )


    # ========================================================
    # 9. THEORETICAL SNR
    # ========================================================

    theoretical_noise_power = (
        noise_power / N
    )

    snr_theoretical[i] = 10 * np.log10(
        signal_power / theoretical_noise_power
    )


# ============================================================
# 10. DISPLAY IMPORTANT VALUES
# ============================================================

print("SNR RESULTS")
print("----------------------------------")

for N in [1, 16, 64, 256]:

    index = N - 1

    print(
        f"N = {N:3d} | "
        f"Experimental SNR = "
        f"{snr_experimental[index]:.2f} dB | "
        f"Theoretical SNR = "
        f"{snr_theoretical[index]:.2f} dB"
    )


# ============================================================
# 11. PLOT SNR EVOLUTION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    N_values,
    snr_experimental,
    linewidth=1.5,
    label="Experimental SNR"
)

plt.plot(
    N_values,
    snr_theoretical,
    linewidth=2,
    linestyle="--",
    label="Theoretical SNR"
)


# ============================================================
# 12. MARK N = 16, 64, 256
# ============================================================

for N in [16, 64, 256]:

    plt.axvline(
        N,
        linestyle=":",
        alpha=0.7
    )

    plt.scatter(
        N,
        snr_experimental[N - 1],
        zorder=5
    )

    plt.text(
        N,
        snr_experimental[N - 1],
        f"  N={N}",
        verticalalignment="bottom"
    )


# ============================================================
# 13. FORMAT
# ============================================================

plt.xlabel("Number of Averaged Trials (N)")
plt.ylabel("SNR (dB)")

plt.title(
    "Experimental vs Theoretical SNR Evolution"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()