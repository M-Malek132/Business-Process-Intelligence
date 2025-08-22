import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Edit Distance Function (Dynamic Programming) ---
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

# Sample event log (case ID, activity, timestamp)
event_log = [
    (1, 'start', '2025-01-01 10:00:00'),
    (1, 'check', '2025-01-01 10:05:00'),
    (1, 'approve', '2025-01-01 10:10:00'),
    (1, 'end', '2025-01-01 10:15:00'),
    (2, 'start', '2025-01-01 11:00:00'),
    (2, 'check', '2025-01-01 11:05:00'),
    (2, 'end', '2025-01-01 11:10:00'),
    (3, 'start', '2025-01-02 10:00:00'),
    (3, 'check', '2025-01-02 10:05:00'),
    (3, 'approve', '2025-01-02 10:10:00'),
    (3, 'finish', '2025-01-02 10:15:00'),  # Drift: "finish" instead of "end"
    (4, 'start', '2025-01-03 10:00:00'),
    (4, 'check', '2025-01-03 10:05:00'),
    (4, 'approve', '2025-01-03 10:10:00'),
    (4, 'finish', '2025-01-03 10:15:00'),  # Drift: "finish" instead of "end"
]

# Convert the event log into a DataFrame
df = pd.DataFrame(event_log, columns=['case_id', 'activity', 'timestamp'])

# Group by case_id and create the event traces
traces = df.groupby('case_id')['activity'].apply(list).tolist()

# Convert all traces into string format (for edit distance comparison)
trace_strings = [' '.join(trace) for trace in traces]

# Compute the pairwise Levenshtein distances (edit distance) using the dynamic programming approach
def compute_edit_distances(traces):
    num_traces = len(traces)
    distance_matrix = np.zeros((num_traces, num_traces))
    
    for i in range(num_traces):
        for j in range(i+1, num_traces):
            distance = edit_distance_dp(traces[i], traces[j])  # Use edit_distance_dp instead of Levenshtein library
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric distance matrix
    return distance_matrix

# Create the distance matrix using edit distance
distance_matrix = compute_edit_distances(trace_strings)

# Perform KMeans clustering to detect concept drift (based on edit distances)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(distance_matrix)  # Using the precomputed distance matrix

# Add the cluster labels to the original DataFrame, aligned with case_id
df['cluster'] = np.nan  # Initialize cluster column
df['cluster'] = df['case_id'].map(dict(zip(df['case_id'].unique(), kmeans.labels_)))

# --- Trace Plot ---
# Visualize the clusters of traces
plt.figure(figsize=(8, 6))
# Define a color map
cmap = plt.get_cmap('viridis')

# Scatter plot for each unique cluster
for cluster in df['cluster'].unique():
    # Filter cases by cluster label
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['case_id'], cluster_data['cluster'], label=f'Cluster {cluster}', cmap=cmap)

# Add a title, labels, and legend
plt.xlabel('Case ID')
plt.ylabel('Cluster')
plt.title('Detected Concept Drift (Clustering of Event Logs using Edit Distance)')
plt.legend(title='Clusters')
plt.show()
# --- Process Model Plot ---
# Create a process model using networkx (directly follow relations)
G = nx.DiGraph()

# Add nodes (activities) to the graph
for activity in df['activity'].unique():
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
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
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
