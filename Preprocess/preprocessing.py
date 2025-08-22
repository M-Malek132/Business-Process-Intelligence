
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
# 4) Data Preprocessing
# =============================================================================

# Extract unique activities and their frequencies
unique_activities = set()  # Use a set to ensure uniqueness

# Iterate over traces to collect all unique activities
for trace in log:
    for event in trace:
        activity_name = event['concept:name']
        unique_activities.add(activity_name)

print(f"Unique event activities in the log: \n{unique_activities}")

# Convert EventLog to DataFrame for further analysis
events_data = []
for trace in log:
    for event in trace:
        activity_name = event['concept:name']
        timestamp = event['time:timestamp']
        case_id = trace.attributes['concept:name']
        events_data.append([case_id, activity_name, timestamp])

# Create a pandas DataFrame for easier manipulation
df = pd.DataFrame(events_data, columns=["Case ID", "Activity", "Timestamp"])

# Check for missing values and duplicates in the DataFrame
print(f"Missing values per column:\n{df.isnull().sum()}")
df.drop_duplicates(inplace=True)

# Check how many duplicate rows are present in the DataFrame
duplicate_rows = df[df.duplicated()]

# Print the number of duplicates
num_duplicates = duplicate_rows.shape[0]
print(f"Number of duplicate rows: {num_duplicates}")
