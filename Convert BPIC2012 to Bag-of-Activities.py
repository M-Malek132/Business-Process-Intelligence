import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
import os

# Load and convert to DataFrame
log = xes_importer.apply(r'raw_datasets\BPI_Challenge_2012.xes.gz')
df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

# Create bag-of-activities per trace
bag_df = df.groupby("case:concept:name")["concept:name"].agg(lambda x: set(x)).reset_index()
all_activities = sorted(df["concept:name"].unique())

# Convert to binary matrix (each activity = 1 if present in trace)
for activity in all_activities:
    bag_df[activity] = bag_df["concept:name"].apply(lambda acts: 1 if activity in acts else 0)

# Optional: add start timestamp of each case for time-based drift detection
start_times = df.groupby("case:concept:name")["time:timestamp"].min().reset_index()
bag_df = pd.merge(bag_df, start_times, on="case:concept:name")
bag_df.rename(columns={"time:timestamp": "start_time"}, inplace=True)

########################################################################################
# Clustering
########################################################################################

from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

X = bag_df[all_activities]  # binary features
k = 4  # try different values and test silhouette manually

km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(X)
bag_df["cluster"] = clusters

########################################################################################
# Visualization
########################################################################################

# Sort by start time to observe process evolution
bag_df_sorted = bag_df.sort_values("start_time")

# Visualize distribution of clusters over time
import seaborn as sns

bag_df_sorted["month"] = pd.to_datetime(bag_df_sorted["start_time"]).dt.to_period("M")
cluster_dist = bag_df_sorted.groupby(["month", "cluster"]).size().unstack(fill_value=0)

# Plot
cluster_dist.plot(kind="bar", stacked=True, figsize=(14, 6))
plt.title("Cluster Distribution Over Time (Bag-of-Activities)")
plt.xlabel("Month")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()

########################################################################################
# Silhouette Score
########################################################################################

from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

distances = pairwise_distances(X, metric="hamming")
score = silhouette_score(distances, clusters, metric="precomputed")
print("Silhouette score:", score)


########################################################################################
# Show Activity Frequency by Cluster
########################################################################################

# Count activity presence per cluster
cluster_summary = bag_df.groupby("cluster")[all_activities].mean().T * 100  # % of cases in cluster with each activity

# Round and sort
summary = cluster_summary.round(1).sort_values(by=list(cluster_summary.columns), ascending=False)
print(summary)

########################################################################################
# Show Example Traces from Each Cluster
########################################################################################

# Join back to full DataFrame
df["cluster"] = df["case:concept:name"].map(bag_df.set_index("case:concept:name")["cluster"])

# Pick 1 random trace per cluster
sampled_cases = df.groupby("cluster")["case:concept:name"].apply(lambda x: x.sample(1)).values

for cid in sampled_cases:
    print(f"\nTrace {cid}:")
    print(df[df["case:concept:name"] == cid]["concept:name"].tolist())

########################################################################################
# Compare Cluster Timelines
########################################################################################

import seaborn as sns
import matplotlib.pyplot as plt

bag_df["cluster"] = bag_df["cluster"].astype(str)
bag_df["start_time"] = pd.to_datetime(bag_df["start_time"])
sns.histplot(data=bag_df, x="start_time", hue="cluster", multiple="stack", bins=30)
plt.title("Clusters Over Time")
plt.xlabel("Case Start Time")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()


########################################################################################
# Compute Activity Presence % per Cluster
########################################################################################

# % of cases in each cluster that include each activity
activity_presence = bag_df.groupby("cluster")[all_activities].mean().T * 100
activity_presence = activity_presence.round(1)



########################################################################################
# Define Heuristic Rules to Label Clusters
########################################################################################


def label_cluster(activity_percentages):
    if activity_percentages.get("A_SUBMITTED", 0) < 30:
        return "No application submitted"
    if activity_percentages.get("O_ACCEPTED", 0) > 70 and activity_percentages.get("W_Complete application", 0) > 50:
        return "Full process with offer acceptance"
    if activity_percentages.get("O_CREATED", 0) < 30:
        return "Incomplete process, skipped offer"
    if activity_percentages.get("W_Assess eligibility", 0) > 60:
        return "Manual assessment-heavy flow"
    return "Generic loan application flow"


########################################################################################
# Apply Labels
########################################################################################

cluster_labels = {}
for cluster_id in activity_presence.columns:
    activity_profile = activity_presence[cluster_id].to_dict()
    label = label_cluster(activity_profile)
    cluster_labels[cluster_id] = label

# Attach labels to bag_df
bag_df["cluster_label"] = bag_df["cluster"].map(cluster_labels)

########################################################################################
# Visualize Case Count by Cluster Label
########################################################################################

import seaborn as sns
sns.countplot(data=bag_df, y="cluster_label", order=bag_df["cluster_label"].value_counts().index)
plt.title("Cluster Label Distribution")
plt.xlabel("Number of Cases")
plt.ylabel("Cluster Label")
plt.show()

########################################################################################
# Explore Dominant Activities
########################################################################################

# To auto-describe a cluster with top activities:

def top_activities(activity_percentages, n=3, threshold=40):
    return [act for act, pct in sorted(activity_percentages.items(), key=lambda x: -x[1]) if pct > threshold][:n]

for cluster_id in activity_presence.columns:
    print(f"\nCluster {cluster_id} Top Activities: {top_activities(activity_presence[cluster_id].to_dict())}")
