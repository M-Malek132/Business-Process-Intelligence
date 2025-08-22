# load_visualize_log.py

import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer

# Get the current working directory (directory where the script is running)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the file from the parent directory
file_path = os.path.join(current_dir, '..', 'raw_datasets', 'BPI_Challenge_2012.xes.gz')

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load log
log = xes_importer.apply(file_path)

print(f"Number of cases: {len(log)}")

print("First trace (case):")
trace = log[0]
for event in log[0]:
    print({
        "Case ID": trace.attributes["concept:name"],
        "Activity": event["concept:name"],
        "Timestamp": event["time:timestamp"]
    })