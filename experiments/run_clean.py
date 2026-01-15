import torch
import sys
import os
# Ensure project root is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from models.cnn import SimpleCNN
from metrics.calibration import expected_calibration_error
from metrics.entropy import predictive_entropy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3

def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=False, download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss / len(trainloader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_logits.append(outputs)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    acc = correct / total
    ece = expected_calibration_error(all_logits, all_labels)
    entropy = predictive_entropy(all_logits)

    print(f"\nClean Test Accuracy: {acc:.4f}")
    print(f"ECE: {ece:.4f}")

    print(f"Predictive Entropy: {entropy:.4f}")

    # Save results as JSON
    import json
    results = {
        "accuracy": acc,
        "ece": ece,
        "entropy": entropy
    }

    with open("experiments/results_clean_cnn.json", "w") as f:
        json.dump(results, f, indent=4)

    # Ensure checkpoints directory exists before saving
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/cnn_clean.pt")


if __name__ == "__main__":
    main()

# Ensure checkpoints directory exists before saving
os.makedirs("checkpoints", exist_ok=True)
