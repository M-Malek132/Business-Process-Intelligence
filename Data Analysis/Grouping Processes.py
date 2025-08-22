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
from datetime import datetime

# =============================================================================
# 2) Configuration / Input
#    - Set the path to the XES log file and verify it exists
# =============================================================================

# Get the current working directory (directory where the script is running)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the file from the parent directory
file_path = os.path.join(current_dir, '..', 'Hospital Data', 'Hospital Billing - Event Log.xes.gz')

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
        timestamp_str = event['time:timestamp']
        timestamp = timestamp_str.timestamp()
        
        # Extract resource information (assuming it's stored in 'org:resource')
        resource = event.get('org:resource', 'Unknown')  # Default to 'Unknown' if resource is not present
        
        data.append((case_id, activity, timestamp, resource))  # Include resource in the tuple


# Create a DataFrame from the event log data
df = pd.DataFrame(data, columns=['case_id', 'activity', 'timestamp', 'resource'])
print((data[0]))

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

# Count the number of traces with length 1
num_traces_length_1 = sum(1 for length in trace_lengths if length == 1)

# Print the result
print(f"Number of traces with length 1: {num_traces_length_1}")

# Print out the trace lengths (for debugging or inspection purposes)
# print(f"\nTrace lengths: {trace_lengths}")
print(trace_lengths[0])
max_length = max(trace_lengths)
mean_length = np.mean(trace_lengths)
print(f'\nmax_length:{max_length}\nmean_length{mean_length}')

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
# 9) Evaluate Clustering Performance (Silhouette Score)
#    - Calculate silhouette scores for the two clustering models to evaluate performance
# =============================================================================

# KMeans silhouette score
kmeans_silhouette = silhouette_score(trace_lengths_reshaped, trace_clusters)
print(f"\nSilhouette Score for KMeans Clustering: {kmeans_silhouette}")

# =============================================================================
# 10) Display the Clusters (Optional Visualization)
#    - Plot a histogram of trace lengths and visualize cluster distribution
# =============================================================================

# Plot a histogram of trace lengths
plt.figure(figsize=(10, 6))
plt.hist(trace_lengths, bins=5, color='skyblue', edgecolor='black')
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
