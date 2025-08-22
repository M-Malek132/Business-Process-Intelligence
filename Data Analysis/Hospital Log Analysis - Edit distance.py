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
# 5) OPTIONAL: EDIT-DISTANCE (LEVENSHTEIN) BETWEEN TRACES
#    - Build a symmetric distance matrix using Levenshtein on stringified traces
#    - Then cluster with Agglomerative using a *precomputed* distance matrix
#      (KMeans is NOT appropriate for precomputed distances)
# =============================================================================

# Dynamic Programming approach to compute edit distance between two strings
def edit_distance_dp(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    
    # Initialize a (len_str1+1) x (len_str2+1) matrix
    dp_matrix = np.zeros((len_str1 + 1, len_str2 + 1), dtype=int)

    # Initialize the base case
    for i in range(len_str1 + 1):
        dp_matrix[i][0] = i
    for j in range(len_str2 + 1):
        dp_matrix[0][j] = j

    # Fill the dp matrix
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp_matrix[i][j] = min(dp_matrix[i - 1][j] + 1,     # Deletion
                                   dp_matrix[i][j - 1] + 1,     # Insertion
                                   dp_matrix[i - 1][j - 1] + cost) # Substitution

    return dp_matrix[len_str1][len_str2]

# Function to compute distance within each chunk and store in HDF5
def compute_distance_chunk(chunk_range, traces, filename, chunk_size):
    num_traces = len(traces)
    with h5py.File(filename, 'a') as h5file:  # Open the HDF5 file for appending
        for i in chunk_range:
            for j in range(i + 1, num_traces):
                distance = edit_distance_dp(' '.join(traces[i]), ' '.join(traces[j]))  # Levenshtein distance
                # Store result in HDF5 file
                h5file['distances'][i, j] = distance
                h5file['distances'][j, i] = distance  # Symmetric distance matrix

# Compute pairwise Levenshtein distances (edit distance) with HDF5 storage
def compute_edit_distances_parallel_hdf5(traces, num_processes=8, chunk_size=1000, filename='distance_matrix.h5'):
    num_traces = len(traces)
    
    # Create a new HDF5 file to store the distances
    with h5py.File(filename, 'w') as h5file:
        # Create an empty dataset for the distance matrix (using chunking to avoid memory overload)
        distance_matrix = h5file.create_dataset('distances', 
                                               shape=(num_traces, num_traces), 
                                               dtype=np.float64, 
                                               chunks=(chunk_size, chunk_size), 
                                               compression='gzip')
        
        # Split the work across multiple processes
        chunk_ranges = [range(i * chunk_size, min((i + 1) * chunk_size, num_traces)) for i in range(num_traces // chunk_size)]
        
        # Start the multiprocessing pool
        with Pool(processes=num_processes) as pool:
            pool.starmap(compute_distance_chunk, [(chunk, traces, filename, chunk_size) for chunk in chunk_ranges])

    print(f"Distance matrix computation complete. Results saved to {filename}.")
    return filename

# Example usage:
# traces = [...]  # Your list of traces, each trace is a list of activities
distance_matrix_file = compute_edit_distances_parallel_hdf5(traces, num_processes=8, chunk_size=1000, filename='distance_matrix.h5')
print(distance_matrix_file.size)

print(distance_matrix_file[0].size)


# Apply KMeans clustering to detect concept drift (based on edit distances)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(distance_matrix_file)  # Using the precomputed distance matrix

# Add the cluster labels to the original DataFrame, aligned with case_id
df['cluster'] = np.nan  # Initialize cluster column
df['cluster'] = df['case_id'].map(dict(zip(df['case_id'].unique(), kmeans.labels_)))

# --- Trace Plot ---
# Visualize the clusters of traces
plt.figure(figsize=(8, 6))
plt.scatter(df['case_id'], df['cluster'], c=df['cluster'], cmap='viridis')
plt.xlabel('Case ID')
plt.ylabel('Cluster')
plt.title('Detected Concept Drift (Clustering of Event Logs using Edit Distance)')
plt.colorbar(label='Cluster')
plt.show()

# =============================================================================
# 6) OPTIONAL: DIRECTLY-FOLLOWS GRAPH (PROCESS MODEL SKETCH)
#    - Builds a simple DFG with edge weights (frequency)
# =============================================================================

# --- Process Model Plot ---
# Create a process model using networkx (directly follow relations)
import networkx as nx

G = nx.DiGraph()

# Add nodes (activities) to the graph
for activity in unique_activities:
    G.add_node(activity)

# Add edges between activities to show flow
for case_id, trace in df.groupby('case_id')['activity']:
    trace_list = trace.tolist()  # Convert series to list
    for i in range(len(trace_list) - 1):
        # Add an edge from trace[i] to trace[i+1]
        G.add_edge(trace_list[i], trace_list[i + 1], weight=1)

# Draw the process model (directly-follow graph)
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # Layout the graph
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

# Edge labels (weights)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title('Process Model (Directly-Follow Graph)')
plt.axis('off')
plt.show()

# Evaluate the quality of the clustering (Silhouette Score)
silhouette_avg = silhouette_score(distance_matrix, kmeans.labels_, metric="precomputed")
print(f'Silhouette Score: {silhouette_avg}')

# Show the cluster centroids (Levenshtein-based)
print("Cluster centroids (Edit Distance):")
print(kmeans.cluster_centers_)
