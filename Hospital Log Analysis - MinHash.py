
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
        data.append((case_id, activity, timestamp))

# Create a DataFrame from the event log data
df = pd.DataFrame(data, columns=['case_id', 'activity', 'timestamp'])
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
print(f"First few k-shingles: \n{list(all_shingles)[:10]}")

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

# =============================================================================
# 8) Apply LSH (Locality Sensitive Hashing) to Generate Candidate Pairs
# =============================================================================

# Initialize the LSH object with a threshold for similarity and the number of permutations for MinHash
threshold = 0.8  # Similarity threshold for considering pairs as candidates
lsh = MinHashLSH(threshold=threshold, num_perm=200)

# Insert each trace's MinHash signature into the LSH index
for idx, minhash in enumerate(minhash_signatures):
    lsh.insert(f"trace_{idx}", minhash)

# Generate candidate pairs: pairs of traces that are similar enough based on MinHash signatures
candidate_pairs = []
for idx1 in range(len(minhash_signatures)):
    for idx2 in range(idx1 + 1, len(minhash_signatures)):
        if lsh.query(minhash_signatures[idx2]):
            candidate_pairs.append((idx1, idx2))

# Print the candidate pairs generated by LSH
print("Candidate pairs generated by LSH:")
for pair in candidate_pairs:
    print(f"Trace {pair[0]} and Trace {pair[1]}")

# =============================================================================
# 9) Optional: Evaluate Candidate Pairs using Jaccard Similarity
# =============================================================================

# Function to compute Jaccard similarity between two sets of shingles
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Calculate Jaccard similarity for the candidate pairs
for pair in candidate_pairs:
    similarity = jaccard_similarity(set(k_shingles[pair[0]]), set(k_shingles[pair[1]]))
    print(f"Jaccard Similarity between Trace {pair[0]} and Trace {pair[1]}: {similarity:.4f}")
