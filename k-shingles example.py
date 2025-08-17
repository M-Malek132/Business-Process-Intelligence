import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample event log
event_log = [
    ['A', 'B', 'C', 'D', 'A', 'B', 'C'],
    ['A', 'B', 'C', 'D', 'A', 'B', 'C'],
    ['A', 'B', 'C', 'D', 'A', 'C'],
    ['A', 'B', 'C', 'A', 'B', 'C'],
    ['A', 'B', 'D', 'A', 'B']
]

def generate_k_shingles(event_log, k):
    shingles = []
    for trace in event_log:
        # Generate k-shingles for each trace
        for i in range(len(trace) - k + 1):
            shingles.append(tuple(trace[i:i+k]))  # Use tuple to make it hashable
    return shingles

def process_discovery(event_log, k):
    # Generate k-shingles
    shingles = generate_k_shingles(event_log, k)
    
    # Count the frequency of each shingle
    shingle_counts = Counter(shingles)
    
    # Convert to DataFrame for better visualization
    shingle_df = pd.DataFrame(shingle_counts.items(), columns=['Shingle', 'Frequency'])
    
    # Sort by frequency
    shingle_df = shingle_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    
    return shingle_df

def apply_clustering_and_pca(shingle_df):
    # Label encode the shingles to convert them into numerical values
    le = LabelEncoder()
    le.fit(shingle_df['Shingle'].astype(str))  # Make sure the shingles are strings
    shingle_df['Encoded'] = le.transform(shingle_df['Shingle'].astype(str))
    
    # Add a new feature for frequency of each shingle
    shingle_df['Frequency'] = shingle_df['Shingle'].map(shingle_df['Shingle'].value_counts())
    
    # Reshape the encoded shingles and frequency into a format suitable for clustering and PCA
    features = shingle_df[['Encoded', 'Frequency']].values
    
    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    shingle_df['Cluster'] = kmeans.fit_predict(features)
    
    # Apply PCA for dimensionality reduction (2 components for visualization)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    # Add PCA components to the DataFrame
    shingle_df['PCA_1'] = pca_result[:, 0]
    shingle_df['PCA_2'] = pca_result[:, 1]
    
    # Normalize the PCA components to the range [0, 1]
    scaler = MinMaxScaler()
    shingle_df[['PCA_1', 'PCA_2']] = scaler.fit_transform(shingle_df[['PCA_1', 'PCA_2']])
    
    return shingle_df, pca_result

# Perform process discovery with k=3
k = 3
discovered_process = process_discovery(event_log, k)

# Apply clustering and PCA to the discovered process patterns
discovered_process, pca_result = apply_clustering_and_pca(discovered_process)

# Display the discovered process patterns with clustering and PCA components
print(discovered_process)

# Plot the PCA results to visualize the clusters with a legend
plt.figure(figsize=(8, 6))
scatter = plt.scatter(discovered_process['PCA_1'], discovered_process['PCA_2'], c=discovered_process['Cluster'], cmap='viridis')

# Add title and labels
plt.title('PCA of Clusters with Discovered Process Patterns')
plt.xlabel('PCA Component 1 (scaled to [0, 1])')
plt.ylabel('PCA Component 2 (scaled to [0, 1])')

# Add a legend using scatter object
plt.legend(*scatter.legend_elements(), title="Clusters")

# Show the plot
plt.show()

# Optionally, save the results to a CSV file
# discovered_process.to_csv('discovered_process_clusters_with_pca.csv', index=False)
# print("Discovered process patterns with clusters and PCA saved to 'discovered_process_clusters_with_pca.csv'.")
