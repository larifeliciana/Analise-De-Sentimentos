"""
This script contains the feature generator class that is used by the sentiment
classification algorithms to represent a review as a feature vector.

Danushka Bollegala.
2012/10/01.

Change Log
----------

2013/09/25: Modified to generate a one line feature vector representing each review. 
            
"""

from collections import defaultdict
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import string
from data import ler

from sklearn.datasets import load_svmlight_file


def savePMImat(coocMatrixFileName, pmiMatrixFileName):
    """
    Loads the co-occurrence matrix, compute PMI and save the
    result to the pmi matrix.
    """
    from MLIB.utils import dmatrix
    M = dmatrix.DMATRIX(SPARSE=True)
    M.read_matrix(coocMatrixFileName)
    pmiMat = M.get_PMI()
    pmiMat.write_matrix(pmiMatrixFileName)
    pass


def compute_LMI(matrixFileName):
    """
    We will first read the co-occurrence matrix from matrixFile Name.
    Next, we will compute the PPMI values for the matrix.
    """
    mat, rowids = load_svmlight_file(matrixFileName)
    (nrows, ncols) = mat.shape
    colTotals = np.zeros(ncols)
    # Bollegala version
    # colTotals = np.zeros(ncols, dtype=DTYPE)
    for j in range(0, ncols):
        colTotals[j] = np.sum(mat[:, j].data)
    N = np.sum(colTotals)
    for i in range(0, nrows):
        row = mat[i, :]
        rowTotal = np.sum(row.data)
        for j in row.indices:
            mat[i, j] = max(0, np.log( (mat[i, j] * N) / (rowTotal * colTotals[j])))
    return mat


def convertToPMI():
    """
    Convert co-occurrence matrices for each domain into PMI matrices.
    This will speed up the subsequent processing because we no longer
    require to compute pmi values during sentiment classification.
    """
    # domains = ["books", "electronics", "dvd", "kitchen"]
    domains = ["books", "electronics"]
    for domain in domains:
        print("Processing domain %s" % domain)
        savePMImat("data/%s/matrix" % domain,
                   "data/%s/matrix.pmi" % domain)
    pass


class FEATURE_GENERATOR:

    def __init__(self):
        # How many instances should be taken as training data.
        self.stopWords = stopwords.words('english')

        # Bolelegala version
        # self.load_stop_words("./stopWords.txt")
        pass

    def load_stop_words(self, stopwords_fname):
        """
        Read the list of stop words and store in a dictionary.
        """
        self.stopWords = []
        F = open(stopwords_fname)
        for line in F:
            self.stopWords.append(line.strip())
        F.close()
        pass

    def is_stop_word(self, word):
        """
        If the word is listed as a stop word in self.stopWords,
        then returns True. Otherwise, returns False.
        """
        return word in self.stopWords

    # def get_tokens(self, line):
    #     """
    #     Return a list of dictionaries, where each dictionary has keys
    #     lemma (lemmatized word), infl (inflections if any), and pos (the POS tag).
    #     The elements in the list are ordered according to their
    #     appearance in the sentence.
    #     """
    #     elements = line.strip().split()
    #     # first token is the face.
    #     tokens = []
    #     for e in elements:
    #         if e == "^_^":
    #             continue
    #         p = e.split("_")
    #         if len(p) != 2:
    #             continue
    #         word = p[0]
    #         if word.find("+") > 0:
    #             # inflection info available.
    #             g = word.split("+")
    #             lemma = g[0]
    #             infl = g[1]
    #         else:
    #             lemma = word
    #             infl = None
    #         pos = p[1]
    #         tokens.append({'lemma': lemma, 'infl': infl, 'pos': pos})
    #         pass
    #     return tokens

    def get_tokens(self, line):
        """
        Return a list of dictionaries, where each dictionary has keys
        lemma (lemmatized word), infl (inflections if any), and pos (the POS tag).
        The elements in the list are ordered according to their
        appearance in the sentence.
        """
        elements = nltk.word_tokenize(line)
        # first token is the face.

        # pos = nltk.pos_tag(elements)
        lemmas = []
        word = WordNetLemmatizer()
        for i in range(len(elements)):
            lemma = word.lemmatize(elements[i].lower())
            if not self.is_stop_word(lemma) and lemma not in string.punctuation and lemma.isalpha():
                lemmas.append(lemma)
        return lemmas

    def get_rating_from_label(self, label):
        """
        Set the rating using the label.
        """
        if label == "positive":
            return "positive"
        elif label == "negative":
            return "negative"
        elif label == "unlabeled":
            return None
        pass

    # Bollegala version
    # def process_file(self, fname, label=None):
    #     """
    #     Open the file fname, generate all the features and return
    #     as a list of feature vectors.
    #     """
    #     feature_vectors = [] #List of feature vectors.
    #     F = open(fname)
    #     line = F.readline()
    #     inReview = False
    #     count = 0
    #     tokens = []
    #     while line:
    #         if line.startswith('^^ <?xml version="1.0"?>'):
    #             line = F.readline()
    #             continue
    #         if line.startswith("<review>"):
    #             inReview = True
    #             tokens = []
    #             line = F.readline()
    #             continue
    #         if inReview and line.startswith("<rating>"):
    #             # Do not uncomment the following line even if you are not
    #             # using get_rating_from_score because we must skip the rating line.
    #             ratingStr = F.readline()
    #             line = F.readline() #skipping the </rating>
    #             continue
    #         if inReview and line.startswith("<Text>"):
    #             while line:
    #                 if line.startswith("</Text>"):
    #                     break
    #                 if len(line) > 1 and not line.startswith("<Text>"):
    #                     curTokens = self.get_tokens(line.strip())
    #                     if curTokens:
    #                         tokens.extend(curTokens)
    #                 line = F.readline()
    #         if inReview and line.startswith("</review>"):
    #             inReview = False
    #             # generate feature vector from tokens.
    #             # Do not use rating related features to avoid overfitting.
    #             fv = self.get_features(tokens, rating=None)
    #             feature_vectors.append((label, fv))
    #             tokens = []
    #             count += 1
    #         line = F.readline()
    #     # write the final lines if we have not seen </review> at the end.
    #     if inReview:
    #         count += 1
    #     F.close()
    #     print(fname, len(feature_vectors))
    #     return feature_vectors

    def process_file(self, fname, label=None):
        """
        Open the file fname, generate all the features and return
        as a list of feature vectors.
        """
        feature_vectors = [] #List of feature vectors.
        F = ler(fname)
        for line in F:
            lemmas = self.get_tokens(line)
            fv = self.get_features(lemmas)
            feature_vectors.append((label, fv))

        print(fname, len(feature_vectors))
        return feature_vectors

    # Bollegala Version
    # def get_features(self, tokens, rating=None):
    #     """
    #     Create a feature vector from the tokens and return it.
    #     """
    #     fv = defaultdict(int)
    #     lemmas = []
    #     # generate unigram features
    #     for token in tokens:
    #         lemma = token["lemma"]
    #         lemmas.append(lemma)
    #         fv[lemma] += 1
    #
    #     # generate bigram features.
    #     for i in range(len(lemmas) - 1):
    #         bigram = "%s__%s" % (lemmas[i], lemmas[i+1])
    #         fv[bigram] += 1
    #     return fv

    def get_features(self, lemmas):
        """
        Create a feature vector from the tokens and return it.
        """
        fv = defaultdict(int)
        # generate unigram and bigram features
        for i in range(len(lemmas)):
            fv[lemmas[i]] += 1
            if i < (len(lemmas) - 1):
                bigram = "%s__%s" % (lemmas[i], lemmas[i + 1])
                fv[bigram] += 1
        return fv


# if __name__ == "__main__":
#     convertToPMI()

