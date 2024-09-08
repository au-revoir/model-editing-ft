import json
from itertools import chain
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from attr_snippets import AttributeSnippets

def get_tfidf_vectorizer(tfidf_path):
    """
    Returns an sklearn TF-IDF vectorizer. See their website for docs.
    Loading hack inspired by some online blog post lol.
    """


    idf_loc, vocab_loc = tfidf_path + "/idf.npy", tfidf_path + "/tfidf_vocab.json"

    idf = np.load(idf_loc)
    with open(vocab_loc, "r") as f:
        vocab = json.load(f)

    class MyVectorizer(TfidfVectorizer):
        TfidfVectorizer.idf_ = idf

    vec = MyVectorizer()
    vec.vocabulary_ = vocab
    vec._tfidf._idf_diag = sp.spdiags(idf, diags=0, m=len(idf), n=len(idf))

    return vec
