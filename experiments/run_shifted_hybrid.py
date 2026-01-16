import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import json
from torch.utils.data import DataLoader

from models.hybrid_qml import HybridQML
from shifts.corruptions import CIFAR10C
from metrics.calibration import expected_calibration_error
from metrics.entropy import predictive_entropy

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16   # IMPORTANT: keep small for hybrid QML

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness"]
SEVERITIES = [1, 2, 3, 4, 5]

CHECKPOINT_PATH = "checkpoints/hybrid_clean.pt"
RESULTS_PATH = "experiments/results_cifar10c_hybrid.json"

# EVALUATION FUNCTION
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    logits_all, labels_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            logits_all.append(logits)
            labels_all.append(y)

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)

    acc = correct / total
    ece = expected_calibration_error(logits_all, labels_all)
    entropy = predictive_entropy(logits_all)

    return acc, ece, entropy

# MAIN
def main():
    print("Running Hybrid QML CIFAR-10-C evaluation")
    print("Device:", DEVICE)

    model = HybridQML().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    results = {}

    for corruption in CORRUPTIONS:
        results[corruption] = {}

        for severity in SEVERITIES:
            print(f"\nEvaluating {corruption} | severity {severity}")

            dataset = CIFAR10C(
                corruption=corruption,
                severity=severity,
                root="./data/cifar10c"
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
            )

            acc, ece, entropy = evaluate(model, loader)

            results[corruption][severity] = {
                "accuracy": acc,
                "ece": ece,
                "entropy": entropy
            }

            print(
                f"→ acc={acc:.4f}, ece={ece:.4f}, entropy={entropy:.4f}"
            )

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\nHybrid CIFAR-10-C evaluation complete.")
    print("Results saved to:", RESULTS_PATH)

if __name__ == "__main__":
    main()