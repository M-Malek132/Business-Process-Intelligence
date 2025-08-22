
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
from datasketch import MinHash, MinHashLSH
 
# =============================================================================
# 2) Configuration / Input
#    - Set the path to the XES log file and verify it exists
# =============================================================================

# Force use of IMf (Inductive Miner for Petri nets)
file_path = r'Hospital Data\Hospital Billing - Event Log.xes.gz'

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

                # Convert the timestamp to epoch time (seconds)
        epoch_time = pd.to_datetime(timestamp).value // 10**9
        
        data.append((case_id, activity, epoch_time))

# Create a DataFrame from the event log data
df = pd.DataFrame(data, columns=['case_id', 'activity', 'epoch_time'])

# Sort each case_id by increasing epoch_time
df = df.sort_values(by=['case_id', 'epoch_time'])

print(df[0:10])

# =============================================================================
# 6) Build Traces (sequence of activities per case)
#    - Group by case_id and collect ordered activity lists
# =============================================================================

# Group by case_id to create traces (sequences of activities)
traces = df.groupby('case_id')['activity'].apply(list).tolist()

# =============================================================================
# 6) Generate k-shingles (k-length subsequences)
# =============================================================================

# Function to generate k-shingles for a trace
def generate_k_shingles(trace, k):
    shingles = []
    for i in range(len(trace) - k + 1):
        shingles.append(tuple(trace[i:i + k]))  # Create tuple to be hashable
    return shingles

# Set the size of the k-shingles
k = 3  # You can adjust this value

# Generate k-shingles for each trace
k_shingles = [generate_k_shingles(trace, k) for trace in traces]

# Flatten the list of shingles and convert to a set to remove duplicates
all_shingles = set([shingle for sublist in k_shingles for shingle in sublist])

# Print first few shingles to verify
# print(f"First few k-shingles: \n{list(all_shingles)[:10]}")
# print(len(all_shingles))

# =============================================================================
# 7) Compute MinHash Signatures (Create fast fingerprints for each trace)
# =============================================================================

from datasketch import MinHash
# Function to compute MinHash signature for a set of shingles (trace)
def compute_minhash_signature(shingles_set, num_hashes=200):
    minhash = MinHash(num_perm=num_hashes)
    for shingle in shingles_set:
        minhash.update(str(shingle).encode('utf8'))  # Update with each shingle
    return minhash

# Compute MinHash signatures for all traces (using k-shingles)
minhash_signatures = [compute_minhash_signature(shingles_set) for shingles_set in k_shingles]
# print(minhash_signatures[0:10])

# =============================================================================
# 8) K-Mode Clustering (for categorical data like process traces)
# =============================================================================
# Transform the activity traces into a suitable format for K-Mode clustering
# Convert the traces into a matrix of activities using the bag of activities approach

# Assuming 'traces' is a list of activity sequences
# Convert activities into a categorical representation (matrix)
label_encoder = LabelEncoder()
encoded_traces = [label_encoder.fit_transform(trace) for trace in traces]  # Encoding each trace

# Now apply K-mode clustering to this encoded data
k_mode = KMeans(n_clusters=5)  # Choose an appropriate number of clusters
k_mode_labels = k_mode.fit_predict(encoded_traces)

# Print results
print("K-Mode Clustering Labels:", k_mode_labels)

# =============================================================================
# 9) Density-Based Clustering (DBSCAN) for detecting clusters based on density
# =============================================================================
# DBSCAN does not require the number of clusters in advance, making it useful for detecting arbitrarily shaped clusters
# Apply DBSCAN to the traces' encoded data
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust the 'eps' and 'min_samples' for your data
dbscan_labels = dbscan.fit_predict(encoded_traces)

# Print results
print("DBSCAN Clustering Labels:", dbscan_labels)

# Optionally, print the cluster sizes
print("Number of clusters from K-Mode:", len(set(k_mode_labels)))
print("Number of clusters from DBSCAN:", len(set(dbscan_labels)))

