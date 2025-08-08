# load_visualize_log.py

import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer

# Force use of IMf (Inductive Miner for Petri nets)

file_path = r'raw_datasets\BPI_Challenge_2012.xes.gz'

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load log
log = xes_importer.apply(file_path)

tree = inductive_miner.apply(log)
gviz = pt_visualizer.apply(tree)
pt_visualizer.view(gviz)