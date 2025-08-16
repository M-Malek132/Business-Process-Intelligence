import pandas as pd
from collections import Counter

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

# Perform process discovery with k=3
k = 3
discovered_process = process_discovery(event_log, k)

# Display the discovered process patterns (frequent k-shingles)
print(discovered_process)

# # Optionally, save the results to a CSV file
# discovered_process.to_csv('discovered_process_patterns.csv', index=False)
# print("Discovered process patterns saved to 'discovered_process_patterns.csv'.")
