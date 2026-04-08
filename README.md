# WGAN-GP for High-Fidelity Financial Transaction Synthesis

## 📌 Project Overview
This repository implements a specialized **Wasserstein GAN with Gradient Penalty (WGAN-GP)** designed to generate high-fidelity synthetic financial transaction data. [cite_start]The architecture is engineered to capture complex, high-dimensional distributions and multivariate relationships inherent in transaction systems while ensuring data privacy. [cite: 1330, 1612]

[cite_start]The model is designed to produce synthetic datasets that maintain a **1:1 distribution match** with original data, providing a robust, anonymous alternative for training downstream machine learning models. [cite: 1221, 1450]

## 🏗️ Repository Structure
* [cite_start]`train_wgan.py`: The core training script utilizing WGAN-GP architecture to optimize Earth Mover’s distance. [cite: 1341, 1607]
* [cite_start]`validate_gan.py`: A comprehensive statistical validation pipeline to verify the fidelity of the generated data. [cite: 1613, 1621]

## ⚙️ Model Architecture & Hyperparameters
### Generator
* [cite_start]**Structure:** 3-layer Dense architecture (2048 -> 1024 -> 512 units). [cite: 1586, 1607]
* [cite_start]**Techniques:** Utilizes `HeNormal` initialization, `LeakyReLU` (alpha=0.2), and `BatchNormalization` for stable gradient flow. [cite: 1607, 1609]
* [cite_start]**Output:** Linear activation to match standardized feature scales. [cite: 1586, 1607]

### Discriminator (Critic)
* [cite_start]**Structure:** 3-layer Dense architecture (512 -> 256 -> 128 units). [cite: 1586, 1608]
* [cite_start]**Regularization:** Employs `Dropout` (0.3) to prevent overfitting. [cite: 1586, 1608]

### Training Parameters
| Parameter | Value |
| :--- | :--- |
| **Latent Dimension** | 128 |
| **Gradient Penalty (GP) Weight** | 30 |
| **Training Ratio** | 5 Critic updates per 1 Generator update |
| **Adam Optimizer (G)** | Learning Rate: 0.0002, beta_1: 0.5 |
| **Adam Optimizer (D)** | Learning Rate: 0.0001, beta_1: 0.5 |

[cite_start][cite: 1587, 1609]

## 🧪 Validation Pipeline
To ensure the synthetic output is statistically indistinguishable from real data, the following tests are implemented in `validate_gan.py`:
1.  [cite_start]**Kolmogorov-Smirnov (KS) Test:** To ensure univariate similarity (Target p-value >= 0.05). [cite: 1614, 1620]
2.  [cite_start]**Wasserstein Distance:** Quantifying the distance between real and synthetic probability distributions. [cite: 1614, 1620]
3.  [cite_start]**Dimensionality Reduction:** **PCA** and **t-SNE** projections to verify the preservation of high-dimensional clusters and patterns. [cite: 1614, 1615, 1620]
4.  [cite_start]**KDE Comparison:** Visualizing probability density overlap across all features. [cite: 1615]
5.  [cite_start]**Correlation Matrix:** Ensuring the multivariate relationships (Pearson coefficients) are preserved within a 0.05 margin. [cite: 1614]

## ⚠️ Data Usage & Privacy
**The original training dataset is not included in this repository.** This project is focused on the implementation and validation of the GAN architecture. Users must provide their own preprocessed transaction data in CSV format as specified in the configuration section of the scripts.
