# WGAN-GP for High-Fidelity Financial Transaction Synthesis

## 📌 Project Overview
This repository implements a specialized **Wasserstein GAN with Gradient Penalty (WGAN-GP)** designed to generate high-fidelity synthetic financial transaction data. The architecture captures complex, high-dimensional distributions and multivariate relationships, providing a robust, privacy-preserving alternative for training machine learning models.

The model achieves a **1:1 distribution match** with original data, ensuring the synthetic output remains statistically indistinguishable from the source.

## 🏗️ Repository Structure
* `train_WGAN.py`: The core training script utilizing WGAN-GP architecture.
* `validate_gan.py`: A comprehensive statistical validation pipeline.

## ⚙️ Model Architecture & Hyperparameters
### Generator
* **Layers:** 3-layer Dense (2048 -> 1024 -> 512 units).
* **Techniques:** `HeNormal` initialization, `LeakyReLU` (alpha=0.2), and `BatchNormalization`.
* **Output:** Linear activation for standardized feature scaling.

### Discriminator (Critic)
* **Layers:** 3-layer Dense (512 -> 256 -> 128 units).
* **Regularization:** `Dropout` (0.3) to ensure stability.

### Training Parameters
| Parameter | Value |
| :--- | :--- |
| **Latent Dimension** | 128 |
| **GP Weight (λ)** | 30 |
| **Critic Updates** | 5 per 1 Generator update |
| **Optimizers** | Adam (G: 0.0002, D: 0.0001) |

## 🧪 Validation Pipeline
The fidelity of the synthetic data is verified in `validate_gan.py` using:
1.  **Kolmogorov-Smirnov (KS) Test:** Targets univariate similarity (p-value ≥ 0.05).
2.  **Wasserstein Distance:** Quantifies probability distribution divergence.
3.  **Visual Projections:** PCA and t-SNE for high-dimensional cluster verification.
4.  **KDE Analysis:** Overlap comparison of probability density functions.

## ⚠️ Data Usage & Privacy
**The training dataset is not included.** Users must provide their own preprocessed numeric transaction data in CSV format as specified in the scripts.
