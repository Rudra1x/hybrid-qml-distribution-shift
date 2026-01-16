import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pennylane as qml
import matplotlib.pyplot as plt
from quantum.circuits import quantum_circuit, n_qubits, n_layers

inputs = [0.1] * n_qubits
weights = [[0.2] * n_qubits for _ in range(n_layers)]

drawer = qml.draw_mpl(quantum_circuit)
fig, ax = drawer(inputs, weights)

# DO NOT use tight_layout for PennyLane circuits
plt.savefig("plots/quantum_circuit_diagram.png", dpi=300, bbox_inches="tight")
plt.close()

print("Quantum circuit diagram saved to plots/quantum_circuit_diagram.png")
