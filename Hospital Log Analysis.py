import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
import os

# Force use of IMf (Inductive Miner for Petri nets)
file_path = r'Hospital Data\Hospital Billing - Event Log.xes.gz'

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load log
log = xes_importer.apply(file_path)

# num_traces = len(log)
# print(f"Number of traces in the log: {num_traces}")

# data = {
#     'case_id': [1, 1, 2, 2, 3, 3],
#     'activity': ['Lab Test Ordered', 'Admission', 'Lab Test Ordered', 'Discharge', 'Admission', 'Lab Test Ordered'],
#     'timestamp': ['2023-08-01 08:00', '2023-08-01 10:00', '2023-08-02 08:00', '2023-08-02 09:00', '2023-08-03 08:00', '2023-08-03 10:00'],
# }

# print(type(data))


# Verify that 'log' is an EventLog
# print(type(log))  # Should print <class 'pm4py.objects.log.obj.EventLog'>

# print(log[0])  # First trace
# print(log[0][0])  # First event in the first trace
# trace = log[0]
# print(trace)
# print(trace.attributes)
# print(type(trace))
# for event in trace:
#     print('\nevents')
#     print(event)
# print(trace.attributes)

# trace.attributes['case:concept:name'] = 'Case 1'

# print(trace.attributes)
# # Convert EventLog to DataFrame using the correct case ID, activity key, and timestamp key
# #df_raw_df = pm4py.format_dataframe(log,
#                                    case_id="caseType",  # Make sure this key matches the case ID in your log
#                                    activity_key="concept:name",  # Activity name key
#                                    timestamp_key="time:timestamp")  # Timestamp key

# # Inspect the DataFrame
# print(type(df_raw_df))  # Should print <class 'pandas.core.frame.DataFrame'>
# print(df_raw_df.head())  # Print the first 5 rows to inspect the data
# print(df_raw_df.shape)  # Get the shape (rows, columns)

# # Extract first trace and work with it
# trace = log[0]

# # Print trace ID and events
# print(f"Trace ID: {trace.attributes.get('concept:name')}")
# for event in trace:
#     print(f"\nEvent: {event}")

# # Modify trace's events or attributes (if needed)
# trace.attributes['new_attribute'] = 'some_value'


# Extract unique event activities across all traces
unique_activities = set()  # Use a set to ensure uniqueness

for trace in log:
    for event in trace:
        activity_name = event['concept:name']
        unique_activities.add(activity_name)  # Add the activity name to the set

print(log[0][0])
print(f"Unique event activities in the log: {unique_activities}")