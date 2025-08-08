import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
import os

# Force use of IMf (Inductive Miner for Petri nets)
file_path = r'raw_datasets\nasa.xes.gz'

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load log
log = xes_importer.apply(file_path)
# Convert to DataFrame
df = log_converter.apply(log)

# Convert to DataFrame
df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

print("=== EVENT LOG SUMMARY ===")
print(f"Number of cases: {df['case:concept:name'].nunique()}")
print(f"Number of events: {len(df)}")
print(f"Unique activities: {df['concept:name'].nunique()}")

# Most frequent activities
print("\nTop 10 activities:")
print(df['concept:name'].value_counts().head(10))

# Trace length statistics
trace_lengths = df.groupby("case:concept:name").size()
print("\nTrace length stats:")
print(trace_lengths.describe())

# Case duration statistics
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
case_durations = df.groupby("case:concept:name")["time:timestamp"].agg(lambda x: x.max() - x.min())
case_durations_hours = case_durations.dt.total_seconds() / 3600
print("\nCase duration stats (hours):")
print(case_durations_hours.describe())

print("\n headers")

print(df.columns.tolist())

print("\n first trace")
trace = log[0]
for event in log[0]:
    print({
        "Case ID": trace.attributes["concept:name"],
        "Activity": event["concept:name"],
        "Timestamp": event["time:timestamp"]
    })