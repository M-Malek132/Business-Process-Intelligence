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

for i in range(10):
    print(f"\n log{i}:\n{log[i]}")

print(len(log))
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
# 7) Bag of Activities (binary vectorization)
#    - Create an index for each unique activity
#    - Encode each trace as a 0/1 vector indicating presence of each activity
# =============================================================================

# # Apply Bag of Activities method
# activity_index = {activity: idx for idx, activity in enumerate(unique_activities)}
# print(f"\nActivity Index: \n{activity_index}")

# def trace_to_bag_of_activities(trace, activity_index):
#     bag = np.zeros(len(activity_index))
#     for activity in trace:
#         bag[activity_index[activity]] = 1
#     return bag

# # Convert traces into Bag of Activities binary vectors
# bag_of_activities = np.array([trace_to_bag_of_activities(trace, activity_index) for trace in traces])

# print(len(bag_of_activities[0]))

# # =============================================================================
# # 1) K-MEANS ON BAG-OF-ACTIVITIES
# #    - Standard KMeans on the binary BoA vectors
# #    - Silhouette uses default (Euclidean) on the feature space
# # =============================================================================

# # 1. K-Means Clustering
# kmeans = KMeans(n_clusters=5, random_state=42)
# kmeans_labels = kmeans.fit_predict(bag_of_activities)

# # Map the cluster labels to the case_id in the original DataFrame
# # Group the original df by 'case_id' to get the trace order
# df['cluster_id'] = df['case_id'].map(dict(zip(df['case_id'].unique(), kmeans_labels)))

# # Evaluate K-Means clustering
# sil_score_kmeans = silhouette_score(bag_of_activities, kmeans_labels)
# print(f"K-Means Silhouette Score: {sil_score_kmeans}")

# =============================================================================
# 2) K-MODES (OPTIONAL)
#    - Best for categorical data; BoA is binary so it can work, but results
#      may differ from KMeans. Uncomment to try.
# =============================================================================

# # 2. K-Mode Clustering (using KModes from kmodes library)
# kmodes = KModes(n_clusters=5, init='Huang', n_init=10, verbose=1)
# kmodes_labels = kmodes.fit_predict(bag_of_activities)

# # Evaluate K-Mode clustering
# sil_score_kmodes = silhouette_score(bag_of_activities, kmodes_labels)
# print(f"K-Mode Silhouette Score: {sil_score_kmodes}")

# =============================================================================
# 3) AGGLOMERATIVE ON BAG-OF-ACTIVITIES
#    - Ward linkage (default) uses Euclidean distances on features
# =============================================================================

# # 3. Agglomerative Clustering
# agg_clustering = AgglomerativeClustering(n_clusters=5)
# agg_labels = agg_clustering.fit_predict(bag_of_activities)

# # Evaluate Agglomerative Clustering
# sil_score_agg = silhouette_score(bag_of_activities, agg_labels)
# print(f"Agglomerative Clustering Silhouette Score: {sil_score_agg}")

# =============================================================================
# 4) 2D VISUALIZATION WITH PCA
#    - Project BoA to 2D for scatter plots of each clustering
# =============================================================================

# # 4. Visualize the results using PCA (2D projection)
# from sklearn.decomposition import PCA
# # Perform PCA for 2D projection
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(bag_of_activities)

# # --- Plotting K-Means Clustering Results with Legend ---
# plt.figure(figsize=(8, 6))

# # Define a color map (you can change this to any other colormap you prefer)
# cmap = plt.get_cmap('viridis')

# # Scatter plot for each unique cluster
# for cluster in np.unique(kmeans_labels):
#     # Filter the points by cluster label
#     cluster_data = reduced_data[kmeans_labels == cluster]
#     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}', cmap=cmap)

# # Add a title, axis labels, and legend
# plt.title("K-Means Clustering Results (PCA 2D)")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.legend(title="Clusters")
# plt.show()

# # # Plotting K-Mode Clustering Results
# # plt.figure(figsize=(8, 6))
# # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmodes_labels, cmap='viridis')
# # plt.title("K-Mode Clustering Results")
# # plt.xlabel("PCA Component 1")
# # plt.ylabel("PCA Component 2")
# # plt.colorbar(label="Cluster Label")
# # plt.show()

# # # Plotting Agglomerative Clustering Results
# # plt.figure(figsize=(8, 6))
# # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=agg_labels, cmap='viridis')
# # plt.title("Agglomerative Clustering Results")
# # plt.xlabel("PCA Component 1")
# # plt.ylabel("PCA Component 2")
# # plt.colorbar(label="Cluster Label")
# # plt.show()

# # =============================================================================
# # 6) OPTIONAL: DIRECTLY-FOLLOWS GRAPH (PROCESS MODEL SKETCH)
# #    - Builds a simple DFG with edge weights (frequency)
# # =============================================================================

# # --- Process Model Plot ---
# # Create a process model using networkx (directly follow relations)
# import networkx as nx

# clusters = df['cluster_id'].unique()

# # Loop over each cluster
# for cluster in clusters:
#     G = nx.DiGraph()
    
#     # Filter data for the current cluster
#     cluster_df = df[df['cluster_id'] == cluster]
    
#     # Add nodes (activities) to the graph
#     unique_activities = cluster_df['activity'].unique()
#     for activity in unique_activities:
#         G.add_node(activity)

#     # Add edges between activities to show flow
#     for case_id, trace in cluster_df.groupby('case_id')['activity']:
#         trace_list = trace.tolist()  # Convert series to list
#         for i in range(len(trace_list) - 1):
#             # Add an edge from trace[i] to trace[i+1]
#             G.add_edge(trace_list[i], trace_list[i + 1], weight=1)

#     # Draw the process model (directly-follow graph) for the current cluster
#     plt.figure(figsize=(10, 8))
#     pos = nx.spring_layout(G, seed=42)  # Layout the graph
#     nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
#     nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
#     nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

#     # Edge labels (weights)
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

#     # Title for each cluster
#     plt.title(f'Process Model (Directly-Follow Graph) - Cluster {cluster}')
#     plt.axis('off')
#     plt.show()

# Assuming 'log' is a pm4py EventLog object
def save_log_to_csv(log, filename="logs_output.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['Case ID', 'Requested Amount', 'Registration Date', 'Event Name', 'Timestamp', 'Resource'])

        # Iterate through the log's cases (log is an EventLog object)
        for case in log:
            case_id = case.attributes['concept:name']  # Extract Case ID
            amount_req = case.attributes.get('AMOUNT_REQ', 'N/A')  # Extract Amount Requested, with fallback
            reg_date = case.attributes.get('REG_DATE', 'N/A')  # Extract Registration Date, with fallback

            # Iterate through the events within a case
            for event in case:
                event_name = event['concept:name']  # Extract Event Name
                timestamp = event['time:timestamp']  # Extract Timestamp
                # Safely access 'org:resource', return None if it doesn't exist
                resource = event.get('org:resource', 'N/A')  # Use 'N/A' as a fallback if the resource is missing

                # Write the row for each event
                writer.writerow([case_id, amount_req, reg_date, event_name, timestamp, resource])

    print(f"Logs have been saved to {filename}")

# Example usage assuming `log` is an EventLog object
save_log_to_csv(log)