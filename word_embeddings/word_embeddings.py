import numpy as np
import torch

def load_embeddings(filename="test.txt"):
    embeddings = {}
    with open(filename) as file:
        for line in file:
            line = line.split(' ')
            word = line[0]
            vec = np.array([float(x) for x in line[1:]], dtype=np.float64)
            embeddings[word] = vec
    return embeddings


def words_to_embeddings(words, embeddings):
    """Given a word sequence words (list of str) and embeddings (dict), returns the embedding for each word if it exists,
    otherwise the word is removed
    
    returns: list of numpy arrays of same length, each array corresponding to a word in the sequence"""

    res = []
    for word in words:
        if word in embeddings:
            res.append(embeddings[word])

    return res

