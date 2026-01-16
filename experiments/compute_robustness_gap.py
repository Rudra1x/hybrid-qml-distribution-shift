import json

CLEAN_RESULTS_FILE = "experiments/results_clean_cnn.json"
SHIFT_RESULTS_FILE = "experiments/results_cifar10c_cnn.json"

with open(CLEAN_RESULTS_FILE, "r") as f:
    clean_results = json.load(f)

with open(SHIFT_RESULTS_FILE, "r") as f:
    shift_results = json.load(f)

clean_acc = clean_results["accuracy"]

print("\nRobustness Gap (ΔR) — CNN Baseline\n")
print("{:<20} {:<15} {:<15} {:<15}".format(
    "Corruption", "Clean Acc", "Severity-5 Acc", "ΔR"
))

print("-" * 65)

robustness_table = {}

for corruption, severities in shift_results.items():
    acc_sev5 = severities["5"]["accuracy"]
    delta_r = clean_acc - acc_sev5

    robustness_table[corruption] = {
        "clean_accuracy": clean_acc,
        "severity_5_accuracy": acc_sev5,
        "robustness_gap": delta_r
    }

    print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(
        corruption, clean_acc, acc_sev5, delta_r
    ))

# Save table
with open("experiments/robustness_gap_cnn.json", "w") as f:
    json.dump(robustness_table, f, indent=4)