import json
import matplotlib.pyplot as plt

RESULTS_FILE = "experiments/results_cifar10c_cnn.json"
SAVE_DIR = "plots/"

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness"]
SEVERITIES = [1, 2, 3, 4, 5]

with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

def plot_metric(metric_name, ylabel, filename):
    plt.figure(figsize=(6, 4))

    for corruption in CORRUPTIONS:
        values = [
            results[corruption][str(sev)][metric_name]
            for sev in SEVERITIES
        ]
        plt.plot(SEVERITIES, values, marker="o", label=corruption)

    plt.xlabel("Corruption Severity")
    plt.ylabel(ylabel)
    plt.xticks(SEVERITIES)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + filename, dpi=300)
    plt.close()

# Accuracy plot
plot_metric(
    metric_name="accuracy",
    ylabel="Accuracy",
    filename="cnn_accuracy_vs_severity.png"
)

# ECE plot
plot_metric(
    metric_name="ece",
    ylabel="Expected Calibration Error (ECE)",
    filename="cnn_ece_vs_severity.png"
)

print("Plots saved to:", SAVE_DIR)