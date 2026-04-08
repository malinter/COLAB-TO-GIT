import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

# Load datasets - Updated to generic redacted paths
original_file = "input_transaction_data.csv"
synthetic_file = "synthetic_transaction_data_10x.csv"

df_original = pd.read_csv(original_file).select_dtypes(include=[np.number])
df_synthetic = pd.read_csv(synthetic_file).select_dtypes(include=[np.number])

# Scale back to original range for comparison
scaler = MinMaxScaler(feature_range=(-1, 1))
df_original_scaled = scaler.fit_transform(df_original)
df_synthetic_scaled = scaler.transform(df_synthetic)

df_original = pd.DataFrame(df_original_scaled, columns=df_original.columns)
df_synthetic = pd.DataFrame(df_synthetic_scaled, columns=df_synthetic.columns)

### 1. Summary Statistics Comparison
print("📊 Summary Statistics Comparison:")
summary_stats_original = df_original.describe()
summary_stats_synthetic = df_synthetic.describe()
display(summary_stats_original)
display(summary_stats_synthetic)

### 2. Kolmogorov-Smirnov Test
print("\n🔬 Running Kolmogorov-Smirnov Test for Distribution Similarity...")
ks_results = {}
for col in df_original.columns:
    ks_stat, p_value = stats.ks_2samp(df_original[col], df_synthetic[col])
    ks_results[col] = {
        'KS Statistic': ks_stat,
        'P-Value': p_value,
        'Anomaly Flag': '⚠️ Possible Anomaly' if p_value < 0.05 and ks_stat > 0.1 else '✔️ OK'
    }
ks_df = pd.DataFrame(ks_results).T
display(ks_df)

### 3. Correlation Matrix Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(df_original.corr(), ax=axes[0], cmap='coolwarm', annot=False)
axes[0].set_title("Original Data Correlation")
sns.heatmap(df_synthetic.corr(), ax=axes[1], cmap='coolwarm', annot=False)
axes[1].set_title("Synthetic Data Correlation")
plt.show()

### 4. Wasserstein Distance Calculation
print("\n📏 Computing Wasserstein Distance...")
wasserstein_results = {col: wasserstein_distance(df_original[col], df_synthetic[col]) for col in df_original.columns}
wasserstein_df = pd.DataFrame.from_dict(wasserstein_results, orient='index', columns=['Wasserstein Distance'])
display(wasserstein_df)

### 5. PCA Visualization
print("\n📉 PCA Projection of Original vs. Synthetic Data...")
pca = PCA(n_components=2)
pca_original = pca.fit_transform(df_original)
pca_synthetic = pca.transform(df_synthetic)
plt.figure(figsize=(8, 6))
plt.scatter(pca_original[:, 0], pca_original[:, 1], alpha=0.5, label='Original Data', color='blue')
plt.scatter(pca_synthetic[:, 0], pca_synthetic[:, 1], alpha=0.5, label='Synthetic Data', color='red')
plt.title("PCA Projection: Original vs Synthetic Data")
plt.legend()
plt.show()

### 6. t-SNE Visualization
print("\n🎭 Running t-SNE Projection for Visualization...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=500)
tsne_original = tsne.fit_transform(df_original)
tsne_synthetic = tsne.fit_transform(df_synthetic)
plt.figure(figsize=(8, 6))
plt.scatter(tsne_original[:, 0], tsne_original[:, 1], alpha=0.5, label='Original Data', color='blue')
plt.scatter(tsne_synthetic[:, 0], tsne_synthetic[:, 1], alpha=0.5, label='Synthetic Data', color='red')
plt.title("t-SNE Projection: Original vs Synthetic Data")
plt.legend()
plt.show()

### 7. KDE Distribution Comparison
print("\n📊 KDE Distribution Comparison...")
num_features = len(df_original.columns)
ncols = 4
nrows = (num_features // ncols) + (num_features % ncols > 0)
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = axes.flatten()
for i, col in enumerate(df_original.columns):
    sns.kdeplot(df_original[col], ax=axes[i], label='Original', color='blue', fill=True)
    sns.kdeplot(df_synthetic[col], ax=axes[i], label='Synthetic', color='red', fill=True)
    axes[i].set_title(f"KDE Plot - {col}")
    axes[i].legend()
plt.tight_layout()
plt.show()

print("\n✅ Validation Pipeline Complete!")
