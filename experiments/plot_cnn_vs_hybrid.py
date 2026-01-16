import json
import matplotlib.pyplot as plt

CNN_FILE = "experiments/results_cifar10c_cnn.json"
HYBRID_FILE = "experiments/results_cifar10c_hybrid.json"
SAVE_DIR = "plots/"

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness"]
SEVERITIES = [1, 2, 3, 4, 5]

with open(CNN_FILE, "r") as f:
    cnn_results = json.load(f)

with open(HYBRID_FILE, "r") as f:
    hybrid_results = json.load(f)

def plot_overlay(metric, ylabel, filename):
    plt.figure(figsize=(6, 4))

    for corruption in CORRUPTIONS:
        cnn_vals = [
            cnn_results[corruption][str(s)][metric] for s in SEVERITIES
        ]
        hybrid_vals = [
            hybrid_results[corruption][str(s)][metric] for s in SEVERITIES
        ]

        plt.plot(
            SEVERITIES, cnn_vals,
            marker="o", linestyle="--",
            label=f"CNN ({corruption})"
        )

        plt.plot(
            SEVERITIES, hybrid_vals,
            marker="s", linestyle="-",
            label=f"Hybrid QML ({corruption})"
        )

    plt.xlabel("Corruption Severity")
    plt.ylabel(ylabel)
    plt.xticks(SEVERITIES)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + filename, dpi=300)
    plt.close()

# Accuracy comparison
plot_overlay(
    metric="accuracy",
    ylabel="Accuracy",
    filename="cnn_vs_hybrid_accuracy.png"
)

# ECE comparison
plot_overlay(
    metric="ece",
    ylabel="Expected Calibration Error (ECE)",
    filename="cnn_vs_hybrid_ece.png"
)

# Entropy comparison
plot_overlay(
    metric="entropy",
    ylabel="Predictive Entropy",
    filename="cnn_vs_hybrid_entropy.png"
)

print("Overlay plots saved to:", SAVE_DIR)