from data_loader import DataLoader
from semantic_clusterer import SemanticClusterer
from lexical_proximity_algorithm import LexicalProximityAlgorithm
from output_manager import OutputManager
from jaccard import JaccardSimilarityCalculator


directory_path = "D:/uni/4 курс/2 семестр/Диплом/similarity/data3"
shingle_size = 5  # Example shingle size

# Initialize DataLoader and load documents as sets of shingles
loader = DataLoader(directory_path, shingle_size)
docs_as_sets = loader.get_docs_as_sets()
print("shingling done")
# Initialize the LexicalProximityAlgorithm with the docs_as_sets
algorithm = LexicalProximityAlgorithm(docs_as_sets, similarity_threshold=0.5)
print("initialized alg")
# Generate MinHash signatures for all documents
signatures = algorithm.generate_minhash_signatures()
print("got signatures")

# Calculate similarities based on the generated MinHash signatures
similar_pairs = algorithm.calculate_similarities(signatures)

# Output the results
for doc1, doc2, sim in similar_pairs:
    if sim > 0.0001:
        print(f"Document {doc1} is similar to Document {doc2} with similarity {sim:.8f}")
