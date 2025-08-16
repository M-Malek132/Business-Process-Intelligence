import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import numpy as np
import os
import hashlib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Force use of IMf (Inductive Miner for Petri nets)
file_path = r'Hospital Data\Hospital Billing - Event Log.xes.gz'

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the event log
log = xes_importer.apply(file_path)

# 1. Min-Hash Implementation
def hash_function(i, value):
    """Returns a hash value for the given element and hash function index"""
    return int(hashlib.md5(f"{i}-{value}".encode('utf-8')).hexdigest(), 16)

def min_hash(set_a, k=100):
    """Generate Min-Hash signature for the given set"""
    signature = []
    for i in range(k):
        # Hash the elements of the set with the i-th hash function
        signature.append(min([hash_function(i, elem) for elem in set_a]))
    return signature

# 2. Convert the event log into sets of activities per case (case-based analysis)
case_activities = {}
for trace in log:
    case_id = trace.attributes['concept:name']
    activities = set()
    for event in trace:
        activities.add(event['concept:name'])
    case_activities[case_id] = activities

# 3. Generate Min-Hash Signatures for All Cases (to represent each case by a vector)
min_hash_signatures = []

for case_id, activities in case_activities.items():
    signature = min_hash(activities, k=100)
    min_hash_signatures.append(signature)

# Convert signatures to a numpy array for clustering
X = np.array(min_hash_signatures)

# 4. Apply KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# 5. Add cluster labels to the cases in the original DataFrame
df = pd.DataFrame(list(case_activities.items()), columns=["case_id", "activities"])
df['cluster'] = df['case_id'].map(lambda case_id: kmeans.labels_[list(case_activities.keys()).index(case_id)])

# 6. Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)

# Get the explained variance ratio to display percentages on the axes
explained_variance = pca.explained_variance_ratio_ * 100

# Plotting the PCA components with clusters as a legend
plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)

# Add a legend using the unique cluster labels
plt.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {i}" for i in range(5)], title="Clusters")

# Add axis labels with percentage of explained variance
plt.xlabel(f"PC1 ({explained_variance[0]:.2f}%)")
plt.ylabel(f"PC2 ({explained_variance[1]:.2f}%)")

plt.title("Clustered Cases based on Min-Hash Signatures")
plt.show()

# 7. Evaluate clustering with silhouette score
sil_score = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {sil_score}")