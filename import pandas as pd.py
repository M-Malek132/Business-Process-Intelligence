import pandas as pd
from kmodes.kmodes import KModes

# Example event log data (replace with your actual data)
data = {
    'case_id': [1, 1, 2, 2, 3, 3],
    'activity': ['Lab Test Ordered', 'Admission', 'Lab Test Ordered', 'Discharge', 'Admission', 'Lab Test Ordered'],
    'timestamp': ['2023-08-01 08:00', '2023-08-01 10:00', '2023-08-02 08:00', '2023-08-02 09:00', '2023-08-03 08:00', '2023-08-03 10:00'],
}

df = pd.DataFrame(data)

# Step 1: Create a list of all unique activities
activities = df['activity'].unique()

# Step 2: Create a binary matrix (bag-of-activities)
binary_matrix = pd.DataFrame(0, index=df['case_id'].unique(), columns=activities)

for _, row in df.iterrows():
    binary_matrix.at[row['case_id'], row['activity']] = 1

# Step 3: Apply K-Mode clustering
kmeans = KModes(n_clusters=2, init='Huang', n_init=10, verbose=1)
clusters = kmeans.fit_predict(binary_matrix)

# Add the clusters to the original DataFrame
binary_matrix['cluster'] = clusters
print(binary_matrix)
