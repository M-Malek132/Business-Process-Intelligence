import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Step 1: Generate a synthetic dataset (e.g., 100 samples, 5 features)
X, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)

# Step 2: Perform PCA on the dataset
pca = PCA()
pca.fit(X)

# Step 3: Calculate the explained variance ratio (eigenvalues)
explained_variance = pca.explained_variance_ratio_

# Calculate the percentage of variance explained by PC1
pc1_percentage = explained_variance[0] * 100
print(f"Percentage of variance explained by PC1: {pc1_percentage:.2f}%")

# Step 4: Create the Scree Plot (for all components)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

# Perform Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=3)
y_pred = agg_clust.fit_predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Agglomerative Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Create a dendrogram to visualize the clustering process
linked = linkage(X, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
