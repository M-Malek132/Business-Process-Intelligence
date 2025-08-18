from datasketch import MinHash

# Create a MinHash object with 200 permutations (hash functions)
num_hashes = 200
minhash = MinHash(num_perm=num_hashes)

# Set A and Set B (example sets)
set_A = {"a", "b", "c", "a"}
set_B = {"b", "c", "d", "e"}

# Add elements of Set A to MinHash object
for element in set_A:
    minhash.update(element.encode('utf8'))

# Save the signature for Set A
set_A_signature = minhash

# Create a new MinHash object for Set B
minhash_B = MinHash(num_perm=num_hashes)

# Add elements of Set B to MinHash object
for element in set_B:
    minhash_B.update(element.encode('utf8'))

# Save the signature for Set B
set_B_signature = minhash_B

# Now, you can compare the signatures of Set A and Set B using the Jaccard Similarity Estimation
# Compute the similarity between the two MinHash signatures
similarity = set_A_signature.jaccard(set_B_signature)

# Print the results
print(f"MinHash Signature for Set A: {set_A_signature.hashvalues}")
print(f"MinHash Signature for Set B: {set_B_signature.hashvalues}")
print(f"Estimated Jaccard Similarity between Set A and Set B: {similarity:.4f}")
