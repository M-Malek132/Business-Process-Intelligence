import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import numpy as np
import os
import Levenshtein as lev
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Force use of IMf (Inductive Miner for Petri nets)
file_path = r'Hospital Data\Hospital Billing - Event Log.xes.gz'

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the event log
log = xes_importer.apply(file_path)

# Extract unique event activities across all traces
unique_activities = set()  # Use a set to ensure uniqueness

for trace in log:
    for event in trace:
        activity_name = event['concept:name']
        unique_activities.add(activity_name)  # Add the activity name to the set

print(f"\nUnique event activities in the log: \n{unique_activities}")

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

# Group by case_id to create traces (sequences of activities)
traces = df.groupby('case_id')['activity'].apply(list).tolist()

# Apply Bag of Activities method
activity_index = {activity: idx for idx, activity in enumerate(unique_activities)}
print(f"\nActivity Index: \n{activity_index}")

def trace_to_bag_of_activities(trace, activity_index):
    bag = np.zeros(len(activity_index))
    for activity in trace:
        bag[activity_index[activity]] = 1
    return bag

# Convert traces into Bag of Activities binary vectors
bag_of_activities = np.array([trace_to_bag_of_activities(trace, activity_index) for trace in traces])

print(len(bag_of_activities[0]))
# Compute the pairwise Levenshtein distances (edit distance)
def compute_edit_distances(traces):
    num_traces = len(traces)
    distance_matrix = np.zeros((num_traces, num_traces))
    
    for i in range(num_traces):
        for j in range(i+1, num_traces):
            distance = lev.distance(' '.join(traces[i]), ' '.join(traces[j]))  # Levenshtein distance
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric distance matrix
    return distance_matrix

# Create the distance matrix using edit distance
distance_matrix = compute_edit_distances(traces)

print(distance_matrix.size)

print(distance_matrix[0].size)
# # Apply KMeans clustering to detect concept drift (based on edit distances)
# kmeans = KMeans(n_clusters=2, random_state=0)
# kmeans.fit(distance_matrix)  # Using the precomputed distance matrix

# # Add the cluster labels to the original DataFrame, aligned with case_id
# df['cluster'] = np.nan  # Initialize cluster column
# df['cluster'] = df['case_id'].map(dict(zip(df['case_id'].unique(), kmeans.labels_)))

# # --- Trace Plot ---
# # Visualize the clusters of traces
# plt.figure(figsize=(8, 6))
# plt.scatter(df['case_id'], df['cluster'], c=df['cluster'], cmap='viridis')
# plt.xlabel('Case ID')
# plt.ylabel('Cluster')
# plt.title('Detected Concept Drift (Clustering of Event Logs using Edit Distance)')
# plt.colorbar(label='Cluster')
# plt.show()

# # --- Process Model Plot ---
# # Create a process model using networkx (directly follow relations)
# import networkx as nx

# G = nx.DiGraph()

# # Add nodes (activities) to the graph
# for activity in unique_activities:
#     G.add_node(activity)

# # Add edges between activities to show flow
# for case_id, trace in df.groupby('case_id')['activity']:
#     trace_list = trace.tolist()  # Convert series to list
#     for i in range(len(trace_list) - 1):
#         # Add an edge from trace[i] to trace[i+1]
#         G.add_edge(trace_list[i], trace_list[i + 1], weight=1)

# # Draw the process model (directly-follow graph)
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G, seed=42)  # Layout the graph
# nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
# nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
# nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

# # Edge labels (weights)
# edge_labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# plt.title('Process Model (Directly-Follow Graph)')
# plt.axis('off')
# plt.show()

# # Evaluate the quality of the clustering (Silhouette Score)
# silhouette_avg = silhouette_score(distance_matrix, kmeans.labels_, metric="precomputed")
# print(f'Silhouette Score: {silhouette_avg}')

# # Show the cluster centroids (Levenshtein-based)
# print("Cluster centroids (Edit Distance):")
# print(kmeans.cluster_centers_)
