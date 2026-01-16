import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pennylane as qml
import matplotlib.pyplot as plt

from models.hybrid_qml import HybridQML
from quantum.circuits import quantum_circuit, n_qubits

# Load trained hybrid model

DEVICE = "cpu"  # drawing only, no need for GPU
model = HybridQML().to(DEVICE)
model.load_state_dict(
    torch.load("checkpoints/hybrid_clean.pt", map_location=DEVICE)
)
model.eval()


# Extract trained quantum weights

trained_weights = model.q_weights.detach().cpu().numpy()


# Use a REAL input example
# (mean of encoder output over one batch is ideal)

# Simple, honest choice: zero-centered input
# (represents normalized latent space used in training)
example_input = [0.0] * n_qubits

# Draw the real circuit
drawer = qml.draw_mpl(quantum_circuit)
fig, ax = drawer(example_input, trained_weights)

plt.tight_layout()
plt.savefig("plots/quantum_circuit_real.png", dpi=300)
plt.close()

print("Real trained quantum circuit saved to plots/quantum_circuit_real.png")