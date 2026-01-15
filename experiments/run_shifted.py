import torch
import sys
import os
# Ensure project root is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from shifts.corruptions import CIFAR10C
from metrics.calibration import expected_calibration_error
from metrics.entropy import predictive_entropy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness"]
SEVERITIES = [1, 2, 3, 4, 5]

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

def main():
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/cnn_clean.pt"))
    model.eval()

    results = {}

    for corruption in CORRUPTIONS:
        results[corruption] = {}
        for severity in SEVERITIES:
            dataset = CIFAR10C(corruption, severity)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            acc, ece, entropy = evaluate(model, loader)

            results[corruption][severity] = {
                "accuracy": acc,
                "ece": ece,
                "entropy": entropy
            }

            print(f"{corruption} | severity {severity} → acc={acc:.4f}, ece={ece:.4f}")

    with open("experiments/results_cifar10c_cnn.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()