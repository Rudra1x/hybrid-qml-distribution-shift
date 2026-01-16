import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

# Box positions
boxes = {
    "data": (0.1, 0.6, "Clean CIFAR-10\nTraining Data"),
    "cnn": (0.35, 0.6, "CNN Baseline\nTraining"),
    "hybrid": (0.35, 0.3, "Hybrid QML\nTraining"),
    "shift": (0.6, 0.6, "Distribution Shift\n(CIFAR-10-C)"),
    "eval": (0.85, 0.45, "Evaluation\nAccuracy, ECE, Entropy"),
}

# Draw boxes
for x, y, text in boxes.values():
    ax.text(
        x, y, text,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"),
        fontsize=10
    )

# Draw arrows
arrows = [
    ((0.18, 0.6), (0.32, 0.6)),
    ((0.18, 0.6), (0.32, 0.3)),
    ((0.45, 0.6), (0.57, 0.6)),
    ((0.45, 0.3), (0.57, 0.6)),
    ((0.72, 0.6), (0.82, 0.45)),
]

for start, end in arrows:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.5)
    )

plt.savefig("plots/procedure_workflow.png", dpi=300, bbox_inches="tight")
plt.close()

print("Procedure workflow diagram saved to plots/procedure_workflow.png")