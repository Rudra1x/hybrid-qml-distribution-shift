import torch

def expected_calibration_error(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(acc - conf)

    return ece.item()