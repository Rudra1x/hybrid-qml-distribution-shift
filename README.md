# Hybrid Quantum–Classical Learning Under Distribution Shift

This repository contains the code accompanying the paper:

**“Hybrid Quantum–Classical Learning Under Distribution Shift”**

## Overview
This work presents a systematic, noise-aware study of how hybrid quantum–classical machine learning models behave under distribution shift, compared against classical deep learning baselines.

We evaluate robustness, uncertainty calibration, and failure modes under multiple corruption-based distribution shifts using CIFAR-10-C.

## Key Findings
- Classical CNNs achieve higher clean accuracy but fail catastrophically and overconfidently under distribution shift.
- Hybrid quantum–classical models degrade more smoothly, exhibiting smaller robustness gaps and more stable uncertainty estimates.
- Quantum measurement-induced uncertainty interacts non-trivially with data distribution shift.

## Repository Structure
- `models/` – CNN and hybrid QML models
- `quantum/` – Variational quantum circuits
- `shifts/` – Distribution shift datasets
- `experiments/` – Training, evaluation, and analysis scripts
- `plots/` – Paper figures
- `paper/` – Manuscript source files

## Datasets
Datasets are not included due to size constraints.

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html  
- CIFAR-10-C: Kaggle

## Reproducibility
All models are trained exclusively on clean data and evaluated under distribution shift without adaptation.

Random seeds and training protocols are fixed for reproducibility.

Author - Rudraksh Sharma
