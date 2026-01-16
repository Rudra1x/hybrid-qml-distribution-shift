import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.hybrid_qml import HybridQML
from metrics.calibration import expected_calibration_error
from metrics.entropy import predictive_entropy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = HybridQML().to(DEVICE)
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
    logits_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            logits_all.append(outputs)
            labels_all.append(labels)

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)

    acc = correct / total
    ece = expected_calibration_error(logits_all, labels_all)
    entropy = predictive_entropy(logits_all)

    print("\nHybrid QML Clean Results")
    print(f"Accuracy: {acc:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Entropy: {entropy:.4f}")

    torch.save(model.state_dict(), "checkpoints/hybrid_clean.pt")

if __name__ == "__main__":
    main()