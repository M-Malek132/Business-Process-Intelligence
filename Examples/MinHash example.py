import hashlib
import numpy as np

# Hash function to be used in Min-Hash
def hash_function(i, value):
    """Returns a hash value for the given element and hash function index"""
    return int(hashlib.md5(f"{i}-{value}".encode('utf-8')).hexdigest(), 16)

# Min-Hash Implementation
def min_hash(set_a, set_b, k=100):
    """Estimate Jaccard similarity between set_a and set_b using Min-Hash"""
    # Initialize Min-Hash signatures for each set
    signature_a = [min([hash_function(i, elem) for elem in set_a]) for i in range(k)]
    signature_b = [min([hash_function(i, elem) for elem in set_b]) for i in range(k)]
    
    # Estimate similarity by comparing how many positions have the same value in the signatures
    similarity = sum([1 if signature_a[i] == signature_b[i] else 0 for i in range(k)]) / k
    return similarity

# Example sets
set_a = {"a", "b", "c", "d"}
set_b = {"a", "c", "d", "e"}

# Estimate similarity using Min-Hash
print("Estimated Jaccard similarity using Min-Hash:", min_hash(set_a, set_b, k=100))


# Example event logs (patient cases as sets of medical procedures)
log_patient_1 = {"checkup", "blood_test", "x_ray", "surgery"}
log_patient_2 = {"checkup", "blood_test", "x_ray", "consultation"}

# Calculate the similarity between the two event logs using Min-Hash
similarity = min_hash(log_patient_1, log_patient_2, k=100)
print("Estimated similarity between patient 1 and patient 2:", similarity)


# Compare Min-Hash similarity with exact similarity (using Edit Distance or Jaccard)
def exact_jaccard(set_a, set_b):
    """Calculate exact Jaccard similarity"""
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

# Compare Min-Hash with exact Jaccard similarity
exact_similarity = exact_jaccard(set_a, set_b)
print("Exact Jaccard similarity:", exact_similarity)
