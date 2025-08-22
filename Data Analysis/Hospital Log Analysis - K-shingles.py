
# =============================================================================
# 1) Imports
# =============================================================================

import pm4py
import h5py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import numpy as np
import os
import Levenshtein as lev
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from multiprocessing import Pool
from scipy.sparse import lil_matrix  # LIL format is efficient for incremental construction

# =============================================================================
# 2) Configuration / Input
#    - Set the path to the XES log file and verify it exists
# =============================================================================

# Get the current working directory (directory where the script is running)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the file from the parent directory
file_path = os.path.join(current_dir, '..', 'raw_datasets', 'BPI_Challenge_2012.xes.gz')

# Normalize the path to avoid issues with different OS path formats
file_path = os.path.normpath(file_path)
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# =============================================================================
# 3) Load Event Log
#    - Read the XES event log into a pm4py EventLog object
# =============================================================================

# Load the event log
log = xes_importer.apply(file_path)

# =============================================================================
# 4) Extract Unique Activities
#    - Iterate all traces and events to collect the set of activity names
# =============================================================================

# Extract unique event activities across all traces
unique_activities = set()  # Use a set to ensure uniqueness

for trace in log:
    for event in trace:
        activity_name = event['concept:name']
        unique_activities.add(activity_name)  # Add the activity name to the set

print(f"\nUnique event activities in the log: \n{unique_activities}")

# =============================================================================
# 5) Convert EventLog -> Tabular Data (DataFrame)
#    - Flatten the event log into rows: (case_id, activity, timestamp)
# =============================================================================

# Convert EventLog to DataFrame for further analysis
data = []
for trace in log:
    case_id = trace.attributes['concept:name']
    for event in trace:
        activity = event['concept:name']
        timestamp = event['time:timestamp']
        data.append((case_id, activity, timestamp))

# Create a DataFrame from the event log data
df = pd.DataFrame(data, columns=['case_id', 'activity', 'timestamp'])

# =============================================================================
# 6) Build Traces (sequence of activities per case)
#    - Group by case_id and collect ordered activity lists
# =============================================================================

# Group by case_id to create traces (sequences of activities)
traces = df.groupby('case_id')['activity'].apply(list).tolist()

# =============================================================================
# 7) Apply K-shingles (k-length subsequences) to each trace
# =============================================================================

# Function to generate k-shingles for each trace
def generate_k_shingles(trace, k):
    shingles = []
    for i in range(len(trace) - k + 1):
        shingles.append(tuple(trace[i:i + k]))  # Create tuple to be hashable
    return shingles

# Set the size of the k-shingles
k = 3  # For example, you can adjust the value of k

# Generate k-shingles for each trace
k_shingles = [generate_k_shingles(trace, k) for trace in traces]

# Flatten the list of shingles and convert to a set to remove duplicates
all_shingles = set([shingle for sublist in k_shingles for shingle in sublist])

# Map each trace to its corresponding shingles
trace_shingles = [set(generate_k_shingles(trace, k)) for trace in traces]

# =============================================================================
# 8) Convert the k-shingles into a feature matrix for clustering
# =============================================================================

# Create a mapping of shingles to indices
shingle_to_index = {shingle: idx for idx, shingle in enumerate(all_shingles)}

# Create a sparse matrix for the feature representation of traces
num_traces = len(traces)
num_shingles = len(all_shingles)
feature_matrix = lil_matrix((num_traces, num_shingles), dtype=int)

# Populate the feature matrix where each trace is represented by a vector of presence (1) or absence (0) of shingles
for trace_idx, shingles_set in enumerate(trace_shingles):
    for shingle in shingles_set:
        feature_matrix[trace_idx, shingle_to_index[shingle]] = 1

# Convert the sparse matrix to a dense matrix for easier handling (optional)
dense_feature_matrix = feature_matrix.toarray()

# =============================================================================
# 9) Apply Clustering (KMeans, Agglomerative, or KModes) on the Feature Matrix
# =============================================================================

# You can now apply clustering algorithms such as KMeans, Agglomerative Clustering, or KModes

# Example: KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Set n_clusters as per your requirements
kmeans_labels = kmeans.fit_predict(dense_feature_matrix)

# Example: Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=5)
agg_labels = agg_clustering.fit_predict(dense_feature_matrix)

# Example: KModes clustering (for categorical data)
kmodes = KModes(n_clusters=5, init='Huang', n_init=10, verbose=1)
kmodes_labels = kmodes.fit_predict(dense_feature_matrix)

# =============================================================================
# 10) Evaluation: Assess clustering results
# =============================================================================

# Evaluate clustering results using Silhouette Score for KMeans and Agglomerative
silhouette_kmeans = silhouette_score(dense_feature_matrix, kmeans_labels)
silhouette_agg = silhouette_score(dense_feature_matrix, agg_labels)

# Print evaluation results
print(f"Silhouette Score for KMeans: {silhouette_kmeans}")
print(f"Silhouette Score for Agglomerative Clustering: {silhouette_agg}")

# =============================================================================
# 11) Visualize Clusters (Optional)
# =============================================================================

# Optionally, you can visualize the clustering results using PCA (Principal Component Analysis) for dimensionality reduction
from sklearn.decomposition import PCA

# Reduce the feature matrix to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(dense_feature_matrix)

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('Clustering Results using KMeans')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.show()
