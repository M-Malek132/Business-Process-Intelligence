import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

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

# Define the activities and create a "Bag of Activities" representation
activities = df['activity'].unique()  # Get all unique activities
activity_index = {activity: idx for idx, activity in enumerate(activities)}

# Convert traces to "Bag of Activities" (binary vector)
def trace_to_bag_of_activities(trace, activity_index):
    bag = np.zeros(len(activity_index))
    for activity in trace:
        bag[activity_index[activity]] = 1
    return bag

# Group by case_id and create the event traces
traces = df.groupby('case_id')['activity'].apply(list).tolist()
print(traces)
# Convert all traces into Bag of Activities vectors
bag_of_activities = np.array([trace_to_bag_of_activities(trace, activity_index) for trace in traces])
print(bag_of_activities)
# Perform KMeans clustering to detect concept drift
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(bag_of_activities)

# Add the cluster labels to the original DataFrame, aligned with case_id
df['cluster'] = np.nan  # Initialize cluster column
df['cluster'] = df['case_id'].map(dict(zip(df['case_id'].unique(), kmeans.labels_)))

# Visualize the clusters
plt.scatter(df['case_id'], df['cluster'], c=df['cluster'], cmap='viridis')
plt.xlabel('Case ID')
plt.ylabel('Cluster')
plt.title('Detected Concept Drift (Clustering of Event Logs)')
plt.show()

# Evaluate the quality of the clustering (Silhouette Score)
silhouette_avg = silhouette_score(bag_of_activities, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')

# Show the cluster centroids
print("Cluster centroids (Bag of Activities):")
print(kmeans.cluster_centers_)

