import pm4py
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.transformation.log_to_data_frame import algorithm as log_to_df
from pm4py.statistics.traces.log import case_statistics
from pm4py.statistics.variants.log import get as variants_get
import pandas as pd

# Force use of IMf (Inductive Miner for Petri nets)

file_path = r'raw_datasets\BPI_Challenge_2012.xes.gz'

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load log
log = xes_importer.apply(file_path)
# Convert to DataFrame
df = log_to_df.apply(log)

print("=== EVENT LOG SUMMARY ===")
print(f"Number of cases: {df['case:concept:name'].nunique()}")
print(f"Number of events: {len(df)}")
print(f"Number of unique activities: {df['concept:name'].nunique()}")

# Most common activities
print("\nTop 10 most frequent activities:")
print(df['concept:name'].value_counts().head(10))

# Trace length statistics
trace_lengths = df.groupby("case:concept:name").size()
print("\nTrace Length Statistics (events per case):")
print(trace_lengths.describe())

# Case duration
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
case_times = df.groupby("case:concept:name")["time:timestamp"]
durations = (case_times.max() - case_times.min()).dt.total_seconds() / 3600  # in hours

print("\nCase Duration Statistics (in hours):")
print(durations.describe())

# Variant analysis
variants_count = variants_get.get_variants_df(log)
print("\nTop 5 Variants:")
print(variants_count.head())

# Optional: show percentage of top 5 variants
top_variants = variants_count.head(5).copy()
top_variants["percent"] = 100 * top_variants["count"] / top_variants["count"].sum()
print("\nTop 5 Variants (with %):")
print(top_variants[["variant", "count", "percent"]])
