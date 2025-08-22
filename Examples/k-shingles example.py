import pandas as pd
from collections import Counter
import numpy as np
from kmodes.kmodes import KModes
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import random

# Original event log
data = [
    ['A', 'B', 'C', 'D', 'A', 'B', 'C'],
    ['A', 'B', 'C', 'D', 'A', 'B', 'C'],
    ['A', 'B', 'C', 'D', 'A', 'C'],
    ['A', 'B', 'C', 'A', 'B', 'C'],
    ['A', 'B', 'D', 'A', 'B'],
    ['B', 'A', 'C', 'D', 'B', 'C'],
    ['C', 'D', 'A', 'B', 'C'],
    ['A', 'C', 'D', 'B', 'A'],
    ['A', 'B', 'C', 'B', 'D'],
    ['B', 'D', 'A', 'C', 'B'],
]

# Function to add small variations to the events
def add_variation(event):
    # Randomly change one event in the sequence
    index_to_change = random.randint(0, len(event) - 1)
    possible_values = ['A', 'B', 'C', 'D']
    event[index_to_change] = random.choice(possible_values)
    return event

# Increase the data size by adding variations
expanded_event_log_with_variation = []
for _ in range(10):  # Increase by 10 times
    for event in data:
        new_event = event.copy()
        new_event = add_variation(new_event)
        expanded_event_log_with_variation.append(new_event)

event_log = expanded_event_log_with_variation

def generate_k_shingles(event_log, k):
    shingles = set()
    for trace in event_log:
        for i in range(len(trace) - k + 1):
            shingles.add(tuple(trace[i:i + k]))  # Use tuple to make it hashable
    return shingles

def create_feature_matrix(event_log, k):
    # Generate global set of shingles across all event logs
    global_shingles = generate_k_shingles(event_log, k)
    global_shingles = sorted(list(global_shingles))  # Sort for consistent column order
    
    # Create feature matrix with frequency count of shingles in each event log
    feature_matrix = []
    for log in event_log:
        # Generate shingles for the current log
        log_shingles = generate_k_shingles([log], k)  # Keep log as list for compatibility
        log_shingle_counts = Counter(log_shingles)  # Count frequency of each shingle
        
        # Create feature vector for the current log based on global shingles
        feature_vector = []
        for shingle in global_shingles:
            feature_vector.append(log_shingle_counts.get(shingle, 0))  # 0 if shingle not in log
        
        feature_matrix.append(feature_vector)
    
    # Convert the feature matrix to a NumPy array for easier handling
    feature_matrix = np.array(feature_matrix)
    return feature_matrix, global_shingles

# Set `k` for k-shingles
k = 2

# Create feature matrix for event logs
features, global_shingles = create_feature_matrix(event_log, k)

# Create a DataFrame for better readability
feature_df = pd.DataFrame(features, columns=[str(shingle) for shingle in global_shingles])

# Display the feature matrix (count of shingles in each event log)
print(feature_df)

def apply_clustering(features):
    # Convert the feature list into a NumPy array
    features_array = np.array(features)
    
    # Apply K-Modes clustering (KMeans in this case)
    km = KMeans(n_clusters=12, random_state=0)
    clusters_kmodes = km.fit_predict(features_array)
    print("\nK-Means Clustering")
    
    # Show number of clusters in K-Means
    num_clusters_kmodes = len(set(clusters_kmodes))
    print(f"Number of clusters in K-Means: {num_clusters_kmodes}")
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    clusters_dbscan = dbscan.fit_predict(features_array)
    print("\nDBSCAN Clustering")
    
    # Show number of clusters in DBSCAN (excluding noise points)
    unique_dbscan = set(clusters_dbscan)
    # Exclude noise points (label -1)
    if -1 in unique_dbscan:
        unique_dbscan.remove(-1)
    num_clusters_dbscan = len(unique_dbscan)
    print(f"Number of clusters in DBSCAN: {num_clusters_dbscan}")
    
    return clusters_kmodes, clusters_dbscan


def plot_clusters(features, clusters_kmodes, clusters_dbscan):
    # PCA for dimensionality reduction (visualization)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    # Plot K-Modes clusters
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    scatter_kmodes = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters_kmodes, cmap='viridis')
    plt.title('K-Modes Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Add a legend for K-Modes clustering
    legend1 = plt.legend(*scatter_kmodes.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)

    # Plot DBSCAN clusters
    plt.subplot(1, 2, 2)
    scatter_dbscan = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters_dbscan, cmap='viridis')
    plt.title('DBSCAN Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Add a legend for DBSCAN clustering
    legend2 = plt.legend(*scatter_dbscan.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend2)

    plt.tight_layout()
    plt.show()

def evaluate_clustering(features, clusters_kmodes, clusters_dbscan):
    # Evaluate K-Modes clustering using Silhouette Score
    print("K-Modes Clustering Evaluation:")
    silhouette_kmodes = silhouette_score(features, clusters_kmodes)
    print(f"Silhouette Score for K-Modes: {silhouette_kmodes:.3f}")
    
    # Evaluate DBSCAN clustering using Silhouette Score (only for non-noise points)
    print("\nDBSCAN Clustering Evaluation:")
    if len(set(clusters_dbscan)) > 1:  # Check if DBSCAN formed more than one cluster
        silhouette_dbscan = silhouette_score(features, clusters_dbscan)
        print(f"Silhouette Score for DBSCAN: {silhouette_dbscan:.3f}")
    else:
        print("DBSCAN found no clusters.")

# Apply clustering (K-Modes and DBSCAN)
clusters_kmodes, clusters_dbscan = apply_clustering(features)

# Plot the clustering results
plot_clusters(features, clusters_kmodes, clusters_dbscan)

# Evaluate clustering performance
evaluate_clustering(features, clusters_kmodes, clusters_dbscan)

# Calculate sum of squared distances (inertia) for different numbers of clusters
inertias = []
for k in range(1, 20):  # Try from 1 to 10 clusters
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(features)
    inertias.append(km.inertia_)

# Plot the elbow graph
plt.plot(range(1, 20), inertias, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# from sklearn.covariance import GraphicalLasso
# from sklearn.preprocessing import StandardScaler
# import networkx as nx
# import matplotlib.pyplot as plt

# # Step 1: Standardize the data (optional, but helps GGM work better)
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # Step 2: Fit the Graphical Lasso model to the data
# model = GraphicalLasso(alpha=0.1)  # Regularization parameter alpha
# model.fit(scaled_features)

# # Step 3: Visualize the graph
# # The precision matrix is the inverse of the covariance matrix
# precision_matrix = model.precision_

# # Build a graph from the precision matrix
# graph = nx.Graph()

# # Add edges where the precision is non-zero
# for i in range(len(global_shingles)):
#     for j in range(i+1, len(global_shingles)):
#         if precision_matrix[i, j] != 0:  # Non-zero precision means a dependency
#             graph.add_edge(global_shingles[i], global_shingles[j], weight=precision_matrix[i, j])

# # Step 4: Draw the graph
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(graph, k=0.15, iterations=20)
# nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
# plt.title('Gaussian Graphical Model - Conditional Dependencies')
# plt.show()
