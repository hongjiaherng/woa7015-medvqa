import nltk
import torch

nltk.download("punkt")
nltk.download("punkt_tab")


def tokenize(q):
    """
    Lowercase the question string `q`, tokenize it using NLTK's word_tokenize, 
    and convert each token to a hash value modulo 5000, so they'll fit within 0-4999.
    Returns a torch tensor of these hash values.
    """
    return torch.tensor([hash(t) % 5000 for t in nltk.word_tokenize(q.lower())])
