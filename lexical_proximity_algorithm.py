import random

class LexicalProximityAlgorithm:
    """
    This class implements the MinHash algorithm to estimate the similarity between documents based on sets of shingles.
    It is designed to compute MinHash signatures for each document and to evaluate pairwise similarities between
    documents using these signatures. A similarity threshold is used to identify and report pairs of documents that
    are considered similar based on their Jaccard similarity estimate.

    The algorithm is particularly useful for large datasets where direct computation of Jaccard similarities would be
    computationally expensive. It provides an efficient probabilistic approach to detect similar documents in a large corpus.

    Attributes:
        docs_as_sets (dict): A dictionary mapping each document ID to its corresponding set of shingles.
        num_hashes (int): The number of hash functions used to compute MinHash signatures, impacting the accuracy of similarity estimates.
        similarity_threshold (float): The minimum similarity score required to consider two documents as similar.
        num_docs (int): The total number of documents being analyzed.
        max_shingle (int): The maximum value for shingle encoding, used in hash function calculations.
        next_prime (int): A prime number larger than max_shingle, used to ensure a good distribution for hash functions.
        coeff_a (list): Random coefficients 'a' used in the hash functions.
        coeff_b (list): Random coefficients 'b' used in the hash functions.
    
    Methods:
        _pick_random_coeffs(): Generates a list of unique random coefficients for the hash functions.
        generate_minhash_signatures(): Computes MinHash signatures for each document using the random hash functions.
        calculate_similarities(signatures): Calculates and returns the pairwise similarities between documents based on their MinHash signatures.
    """

    def __init__(self, docs_as_sets, num_hashes=100, similarity_threshold=0.5):
            """
            Initializes the LexicalProximityAlgorithm with a dictionary of documents and their shingle sets.
            
            Parameters:
                docs_as_sets (dict): Dictionary with document IDs as keys and sets of shingles as values.
                num_hashes (int): Number of hash functions to use in the MinHash algorithm.
                similarity_threshold (float): The threshold for considering documents as similar.
            """
            self.docs_as_sets = docs_as_sets
            self.num_hashes = num_hashes
            self.similarity_threshold = similarity_threshold
            self.num_docs = len(docs_as_sets)
            self.max_shingle = 2**32 - 1
            self.next_prime = 4294967311
            self.coeff_a = self._pick_random_coeffs()
            self.coeff_b = self._pick_random_coeffs()
    
    def _pick_random_coeffs(self):
        """
        Helper method to generate a list of unique random coefficients for the hash functions used in MinHash.
        Our random hash function will take the form of: 
        h(x) = (a*x + b) % c
        Where 'x' is the input value, 'a' and 'b' are random coefficients, 
        and 'c' is a prime number just greater than maxShingleID.
        Returns:
            list: A list of unique random integers.
        """
        rand_set = set()
        while len(rand_set) < self.num_hashes:
            rand_index = random.randint(0, self.max_shingle)
            if rand_index not in rand_set:
                rand_set.add(rand_index)
        return list(rand_set)

    def generate_minhash_signatures(self):
        """
        Generates MinHash signatures for all documents based on their shingle sets.
        
        Returns:
            dict: A dictionary with document IDs as keys and their MinHash signatures as values.
        """
        signatures = []
        for doc_id in self.docs_as_sets:
            shingleIDSet = self.docs_as_sets[doc_id]
            signature = []
            for i in range(0,self.num_hashes):
                min_hash = self.next_prime + 1
                for shingleID in shingleIDSet:
                    hash_code = (self.coeff_a[i] * shingleID + self.coeff_b [i]) % self.next_prime
                    if hash_code < min_hash:
                        min_hash = hash_code
                signature.append(min_hash)
            signatures.append(signature)
        return signatures

    def calculate_similarities(self, signatures):
        """
        Calculates similarities between all pairs of documents based on their MinHash signatures.
        
        Returns:
            list: A list of tuples (doc_id1, doc_id2, similarity) for document pairs with similarities.
        """
        #doc_ids = list(signatures.keys())
        similarities = []
        for i in range(self.num_docs):
            sig1 = signatures[i]
            for j in range(i + 1, self.num_docs):
                sig2 = signatures[j]
                count = 0
                for k in range(0, self.num_hashes):
                    count += (sig1[k] == sig2[k])
                sim = count / self.num_hashes
                similarities.append((i, j, sim))
        return similarities