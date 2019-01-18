import numpy as np
import pandas as pd
from data import salvar, ler, dataset
from collections import defaultdict
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


def tokenize(t):
    import string
    from nltk.corpus import stopwords
    from collections import Counter
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    stop = stopwords.words('english')
    x = []
    t = word_tokenize(t)
    for a in t:
        a = a.lower()
        count = Counter(a)
        if a not in stop and a not in string.punctuation and a.isalpha() and len(a) > 2 and len(count) > 1:
            x.append(WordNetLemmatizer().lemmatize(a))

    return x


def normalize(base):
    from sklearn import preprocessing
    cols = list(base.columns)
    if cols[-1] == 'label' or cols[-1] == 'class_label':
        label = cols.pop()
        new_base = preprocessing.normalize(base[cols])
        labels = pd.DataFrame(base[label], columns=[label])
        new_base = pd.DataFrame(new_base, columns=cols)
        return new_base.join(labels)
    else:
        new_base = preprocessing.normalize(base[cols])
        new_base = pd.DataFrame(new_base, columns=cols)
        return new_base


def filter_by_pos(x):
    if 'NN' in x[1] or 'VB' in x[1] or 'JJ' in x[1] or "RB" in x[1]:
        return True
    else:
        return False


def get_features(sentence):
    import nltk
    features = defaultdict(int)
    tokens = tokenize(sentence)
    pos = nltk.pos_tag(tokens)
    new_pos = list(map(lambda x: x[0], list(filter(filter_by_pos, pos))))

    for i in range(len(new_pos)):
        features[new_pos[i]] += 1
        if i < (len(new_pos) - 1):
            bigram = "%s__%s" % (new_pos[i], new_pos[i+1])
            features[bigram] += 1

    return features


def get_all_features(data):
    dicts = []
    print(len(data))
    for d in data:
        dicts.append(get_features(d))

    return dicts


class SpectralFeatureAlignment:
    def __init__(self, nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma):
        self.nclusters = nclusters
        self.nDI = nDI
        self.coocTh = coocTh
        self.sourceFreqTh = sourceFreqTh
        self.targetFreqTh = targetFreqTh
        self.gamma = gamma
        self.U = None
        self.DS = None

    @staticmethod
    def break_data_label(base):
        columns = list(base.columns)
        label = columns.pop()
        b = base.copy()
        labels = b.pop(label)

        return b, labels

    @staticmethod
    def threshold(h, t):
        p = {}
        for (key, val) in h.items():
            if val > t:
                p[key] = val
        del h
        return p

    def spectral_alignment(self, source, target):
        print('initialized spectral')
        if isinstance(source, pd.DataFrame) and isinstance(target, pd.DataFrame):
            source = source['text']
            target = target['text']

        dict_source, vocab_source = self.get_features_vocab(source, self.sourceFreqTh)
        dict_target, vocab_target = self.get_features_vocab(target, self.targetFreqTh)
        print('features acquired')

        vocab = vocab_source.copy()
        for w in vocab_target:
            src = 0
            tar = 0
            if w in vocab_source:
                src = vocab_source[w]
            if w in vocab_target:
                tar = vocab_target[w]
            vocab[w] = src + tar
        print('%d vocabulary length' % len(vocab))

        corr = {}
        self.correlation(corr, dict_source)
        self.correlation(corr, dict_target)

        self.threshold(corr, self.coocTh)

        pivots = set(vocab_source.keys()).intersection(set(vocab_target.keys()))
        print('%d pivots length' % len(pivots))

        C = {}
        N = sum(vocab.values())
        for pivot in pivots:
            C[pivot] = 0.0
            for w in vocab_source:
                val = self.getVal(pivot, w, corr)
                C[pivot] += 0 if (val < self.coocTh) else self.getPMI(val, vocab[w], vocab[pivot], N)
            for w in vocab_target:
                val = self.getVal(pivot, w, corr)
                C[pivot] += 0 if (val < self.coocTh) else self.getPMI(val, vocab[w], vocab[pivot], N)

        print("pivots MI calculated")

        pivotList = sorted(C.items(), key=lambda x: x[1])
        DI = []
        for (w, v) in pivotList[:self.nDI]:
            DI.append(w)

        DS = set(vocab_source.keys()).union(set(vocab_target.keys())) - pivots
        DS = list(DS)

        nDS = len(DS)
        nDI = len(DI)

        self.DS = {}
        for i in range(nDS):
            self.DS[DS[i]] = i
        salvar('DS_list.ig', self.DS)
        print("Total no. of domain specific features =", len(DS))

        M = np.zeros((nDS, nDI), dtype=np.float)
        for i in range(nDS):
            for j in range(nDI):
                M[i, j] = self.getVal(DS[i], DI[j], corr)
        print('Correlation Matrix computed')

        nV = len(vocab.keys())

        A = self.affinity_matrix(M, len(DS), nV)
        print("Affinity Matrix computed")

        L = self.laplacian_matrix(A, nV)
        print('Laplacian Matrix computed')

        U = self.apply_svd(L, self.nclusters)

        self.U = U
        salvar('U.ig', U)

        print('Done')

        return U

    @staticmethod
    def getVal(x, y, M):
        """
        Returns the value of the element (x,y) in M.
        """
        if (x, y) in M.keys():
            return M[(x, y)]
        elif (y, x) in M.keys():
            return M[(y, x)]
        else:
            return 0
        pass

    @staticmethod
    def getPMI(n, x, y, N):
        import math
        """
        Compute the weighted PMI value.
        """
        v = (float(n) * float(N)) / (float(x) * float(y))
        if v <= 0:
            pmi = 0
        else:
            pmi = math.log(v)
        res = pmi * (float(n) / float(N))
        return 0 if res < 0 else res

    def functionDS(self, xi):
        features = get_features(xi)
        features = list(features.keys())
        x = np.zeros((1, len(self.DS)))
        for f in features:
            if f in self.DS:
                x[0, self.DS[f]] = 1

        return x

    @staticmethod
    def correlation(M, data):
        for d in data:
            d = list(d.keys())
            n = len(d)
            for i in range(n):
                for j in range(i + 1, n):
                    pair = (d[i], d[j])
                    rpair = (d[j], d[i])
                    if pair in M:
                        M[pair] += 1
                    elif rpair in M:
                        M[rpair] += 1
                    else:
                        M[pair] = 1

        return M

    @staticmethod
    def get_features_vocab(dt, t):
        dicts = get_all_features(dt)
        print('features extrated.')
        vocab = defaultdict(int)

        for d in dicts:
            for key, val in d.items():
                vocab[key] += val
        print('%d words in vocabulary' % len(vocab))

        vocab = SpectralFeatureAlignment.threshold(vocab, t)
        print('%d words after thresholding' % len(vocab))

        return dicts, vocab

    @staticmethod
    def affinity_matrix(M, MminusL, nV):
        Mt = M.transpose()
        A = np.zeros((nV, nV), dtype=np.float)

        Mti = 0
        Mtj = 0
        Mi = 0
        Mj = 0
        for i in range(nV):
            for j in range(nV):
                if i > MminusL - 1 and j < MminusL:
                    A[i][j] = Mt[Mti, Mtj]
                    Mtj += 1
                if i < MminusL and j > MminusL - 1:
                    A[i][j] = M[Mi, Mj]
                    Mj += 1
            if i > MminusL - 1:
                Mti += 1
            Mi += 1
            Mtj = 0
            Mj = 0

        return A

    @staticmethod
    def laplacian_matrix(A, nV):
        D = np.zeros((nV, nV), dtype=np.float)
        for i in range(nV):
            D[i][i] = np.sum(A[i, :])

        for i in range(nV):
            for j in range(nV):
                if D[i, j] != 0:
                    D[i, j] = 1 / np.sqrt(D[i, j])

        L = (D.dot(A)).dot(D)

        return L

    @staticmethod
    def apply_svd(L, k):
        from sparsesvd import sparsesvd
        from scipy.sparse import csc_matrix
        _, eigenvectors = np.linalg.eig(L)
        U, _, _ = sparsesvd(csc_matrix(eigenvectors), k)

        return U.transpose()

    def apply_feature_align(self, x):
        if self.U is not None:
            y = x.dot(self.U[0:len(self.DS), :])
            for i in range(self.nclusters):
                y[0, i] = self.gamma * y[0, i]

            return y[0]
        else:
            raise ValueError("U is None, run spectral_alignment before")

    def transform_data(self, dat):
        dt = []
        for xi in dat:
            x_line = self.functionDS(xi)
            x_line = self.apply_feature_align(x_line)
            dt.append(x_line)

        return dt


src_data, src_label = dataset('movies.pk')
tar_data, tar_label = dataset('cds.pk')


nclusters = 100
nDI = 500
coocTh = 10
sourceFreqTh = 20
targetFreqTh = 20
gamma = 0.6
spec = SpectralFeatureAlignment(nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma)
spec.spectral_alignment(src_data, tar_data)

train = spec.transform_data(src_data)
test = spec.transform_data(tar_data)

s = LinearSVC()
s.fit(train, src_label)
predicts = s.predict(test)
print(accuracy_score(tar_label, predicts))
