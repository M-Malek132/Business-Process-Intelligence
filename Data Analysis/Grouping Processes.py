# Grouping Processes into 4

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
import csv

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
print(df[0])
# =============================================================================
# 6) Build Traces (sequence of activities per case)
#    - Group by case_id and collect ordered activity lists
# =============================================================================

# Group by case_id to create traces (sequences of activities)
traces = df.groupby('case_id')['activity'].apply(list).tolist()

# =============================================================================
# 7) Calculate the Length of Each Trace
#    - Determine the length (number of activities) of each trace
# =============================================================================

# Calculate the length of each trace (sequence of activities)
trace_lengths = [len(trace) for trace in traces]

# Print out the trace lengths (for debugging or inspection purposes)
# print(f"\nTrace lengths: {trace_lengths}")
print(trace[0])
# =============================================================================
# 8) KMeans Clustering of Traces Based on Length
#    - Use KMeans to cluster traces based on their length
# =============================================================================

# Reshape the data for KMeans (we need a 2D array for KMeans)
trace_lengths_reshaped = np.array(trace_lengths).reshape(-1, 1)

# Initialize KMeans with 4 clusters (as per the requirement)
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model
kmeans.fit(trace_lengths_reshaped)

# Get the cluster labels
trace_clusters = kmeans.labels_

# Print out the cluster assignments for each trace
print(f"\nCluster assignments based on length:\n{trace_clusters}")

# =============================================================================
# 9) Agglomerative Clustering of Traces Based on Length
#    - Use AgglomerativeClustering for a hierarchical approach
# =============================================================================

# Initialize AgglomerativeClustering with 4 clusters
agg_clustering = AgglomerativeClustering(n_clusters=4)

# Fit the model
agg_labels = agg_clustering.fit_predict(trace_lengths_reshaped)

# Print out the cluster assignments from AgglomerativeClustering
print(f"\nAgglomerative clustering cluster assignments:\n{agg_labels}")

# =============================================================================
# 10) Evaluate Clustering Performance (Silhouette Score)
#    - Calculate silhouette scores for the two clustering models to evaluate performance
# =============================================================================

# KMeans silhouette score
kmeans_silhouette = silhouette_score(trace_lengths_reshaped, trace_clusters)
print(f"\nSilhouette Score for KMeans Clustering: {kmeans_silhouette}")

# AgglomerativeClustering silhouette score
agg_silhouette = silhouette_score(trace_lengths_reshaped, agg_labels)
print(f"Silhouette Score for AgglomerativeClustering: {agg_silhouette}")

# =============================================================================
# 11) Display the Clusters (Optional Visualization)
#    - Plot a histogram of trace lengths and visualize cluster distribution
# =============================================================================

# Plot a histogram of trace lengths
plt.figure(figsize=(10, 6))
plt.hist(trace_lengths, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Trace Lengths")
plt.xlabel("Trace Length (Number of Activities)")
plt.ylabel("Frequency")
plt.grid(True)

# Plot KMeans clusters
plt.figure(figsize=(10, 6))
plt.scatter(range(len(trace_lengths)), trace_lengths, c=trace_clusters, cmap='viridis')
plt.title("KMeans Clustering of Traces by Length")
plt.xlabel("Trace Index")
plt.ylabel("Trace Length")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

# =============================================================================
# 12) Save Cluster Information to CSV
#    - Save the trace clusters to a CSV file
# =============================================================================

# Create a DataFrame with case_id and corresponding cluster label
cluster_df = pd.DataFrame({
    'case_id': df['case_id'].unique(),
    'cluster': trace_clusters
})

# Save to CSV
output_file = os.path.join(current_dir, 'trace_clusters.csv')
cluster_df.to_csv(output_file, index=False)

print(f"\nTrace clusters saved to: {output_file}")
