import torch
import torch.nn as nn
import pennylane as qml

from quantum.circuits import quantum_circuit, n_qubits, n_layers

class HybridQML(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Classical encoder (small CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_embed = nn.Linear(32, n_qubits)

        # Quantum weights
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits)
        )

        # Classical readout
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_embed(x)

        # Normalize for AngleEmbedding
        x = torch.tanh(x)

        q_out = []
        for i in range(x.shape[0]):
            q_res = quantum_circuit(x[i], self.q_weights)
            q_out.append(torch.stack(q_res))

        q_out = torch.stack(q_out)
        return self.classifier(q_out.float())