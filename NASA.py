from pm4py.objects.log.importer.xes import factory as xes_importer

# Load the XES file (replace with your actual file path)
log = xes_importer.apply(r'raw_datasets\NASA.xes')


# Check number of traces (cases)
print("Number of cases:", len(log))
