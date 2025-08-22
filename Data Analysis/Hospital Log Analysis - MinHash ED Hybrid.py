import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import numpy as np
import os
import hashlib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import Levenshtein as lev
from scipy.sparse import lil_matrix

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

def min_hash(set_a, set_b, k=100):
    """Estimate Jaccard similarity between set_a and set_b using Min-Hash"""
    signature_a = [min([hash_function(i, elem) for elem in set_a]) for i in range(k)]
    signature_b = [min([hash_function(i, elem) for elem in set_b]) for i in range(k)]
    
    similarity = sum([1 if signature_a[i] == signature_b[i] else 0 for i in range(k)]) / k
    return similarity

# 2. Convert the event log into sets of activities per case (case-based analysis)
case_activities = {}
for trace in log:
    case_id = trace.attributes['concept:name']
    activities = set()
    for event in trace:
        activities.add(event['concept:name'])
    case_activities[case_id] = activities

# 3. Apply Min-Hash to quickly identify similar logs
case_ids = list(case_activities.keys())

# Create a sparse matrix for the similarity values (using lil_matrix for efficient row updates)
similarity_matrix = lil_matrix((len(case_ids), len(case_ids)))

# Applying Min-Hash to filter similar cases
min_hash_threshold = 0.5  # Define a threshold for similarity
similar_cases = []

# Iterate through the cases and calculate Min-Hash similarity
for i in range(len(case_ids)):
    for j in range(i + 1, len(case_ids)):
        case_i = case_activities[case_ids[i]]
        case_j = case_activities[case_ids[j]]
        similarity = min_hash(case_i, case_j, k=100)
        
        if similarity >= min_hash_threshold:
            # Store the similarity in the sparse matrix
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
            similar_cases.append((case_ids[i], case_ids[j], similarity))

# Now, we have a reduced set of similar cases that will undergo Edit Distance comparison
print(f"Number of similar case pairs based on Min-Hash: {len(similar_cases)}")

# 4. Apply Edit Distance to the filtered set of similar logs (Min-Hash reduced set)
def edit_distance_case_pairs(similar_cases):
    """Apply Edit Distance on the reduced set of similar logs"""
    final_similarities = []
    for case_i, case_j, _ in similar_cases:
        set_i = case_activities[case_i]
        set_j = case_activities[case_j]
        
        # Calculate the exact similarity using Edit Distance
        dist = lev.distance(str(set_i), str(set_j))  # Convert sets to strings and calculate distance
        max_len = max(len(set_i), len(set_j))
        similarity = 1 - (dist / max_len)  # Normalize the similarity based on the maximum length
        final_similarities.append((case_i, case_j, similarity))
    return final_similarities

# Get the final set of similarities using Edit Distance
final_similarities = edit_distance_case_pairs(similar_cases)

# 5. Evaluate Performance (Hybrid vs Edit Distance alone)
# First, calculate the Edit Distance similarity between all pairs
def full_edit_distance():
    """Compute the exact Edit Distance similarity between all pairs of cases"""
    full_similarities = []
    for i in range(len(case_ids)):
        for j in range(i + 1, len(case_ids)):
            case_i = case_activities[case_ids[i]]
            case_j = case_activities[case_ids[j]]
            
            dist = lev.distance(str(case_i), str(case_j))
            max_len = max(len(case_i), len(case_j))
            similarity = 1 - (dist / max_len)
            full_similarities.append((case_ids[i], case_ids[j], similarity))
    return full_similarities

# Get the full set of similarities using Edit Distance
full_similarities = full_edit_distance()

# Compare the results (Optional: Implement a performance comparison logic)
# For now, print the size reduction achieved by Min-Hash filtering
print(f"Original number of case pairs: {len(case_ids) * (len(case_ids) - 1) / 2}")
print(f"Reduced number of case pairs after Min-Hash filtering: {len(similar_cases)}")

# 6. Clustering the cases based on Min-Hash and Edit Distance similarities
# Use KMeans on the reduced similarity matrix to cluster cases
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(similarity_matrix[:len(similar_cases)])  # Use the filtered similarity matrix

# Add cluster labels to the cases
df['cluster'] = df['case_id'].map(lambda case_id: kmeans.labels_[case_ids.index(case_id)])

# Visualize the clusters using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(similarity_matrix)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("Clustered Cases based on Min-Hash Similarity")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.show()

# 7. Silhouette Score for clustering evaluation
sil_score = silhouette_score(similarity_matrix, kmeans.labels_)
print(f"Silhouette Score (after applying Min-Hash and Edit Distance): {sil_score}")


