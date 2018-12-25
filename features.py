
from sklearn import feature_extraction
from sklearn import feature_selection
import delta
import numpy as np

def feature_extraction_methods(treino, teste, tipo, stopwords, smooth):
    if stopwords:
        stopwords = 'english'

    pre = []
    if tipo is 'tfidf':
        pre = feature_extraction.text.TfidfVectorizer(stop_words=stopwords, smooth_idf=smooth)
    elif tipo is 'idf':
        pre = feature_extraction.text.TfidfVectorizer(stop_words=stopwords, binary=True)
    elif tipo is 'counter':
        pre = feature_extraction.text.CountVectorizer(stop_words=stopwords)
    elif tipo is 'binario':
        pre = feature_extraction.text.CountVectorizer(stop_words=stopwords)
    elif tipo is 'delta':
        pre = delta.TfidfVectorizer()

    return pre.fit_transform(treino), pre.transform(teste)#, pre.get_feature_names()


def feature_selection_methods(treino, classes, teste, metodo, k, doc_treino, doc_teste): ##precisa dos dados originais para selectionar com o tfidf e o delta
    stopwords = 'english'
    # Selecionar os k melhores ranqueados de acordo com o método
    if metodo is "chi":
        funct = feature_selection.SelectKBest(feature_selection.chi2, k=k)
    elif metodo is "anova":
        funct = feature_selection.SelectKBest(feature_selection.f_classif, k=k)
    elif metodo is "tfidf":
        if stopwords:
            stopwords = 'english'
        pre = feature_extraction.text.TfidfVectorizer(stop_words=stopwords, binary=True)
        tr = pre.fit_transform(doc_treino)
        te = pre.transform(doc_teste)
        selectKbest(pre, treino,k)

    elif metodo is "delta":
        if stopwords:
            stopwords = 'english'
        pre = feature_extraction.text.TfidfVectorizer(stop_words=stopwords, binary=True)
        tr = pre.fit_transform(doc_treino)
        te = pre.transform(doc_teste)
        selectKbest(pre, treino, k)
        
    else:
        return treino, teste

    treino = funct.fit_transform(treino, classes)
    teste = funct.transform(teste)

 #   cols = funct.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
  #  for i in cols:
   #     print(features[i])

    return treino, teste




def selectKbest(vectorizer, tfidf_result,k):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores = [i[0] for i in sorted_scores]
    return sorted_scores[0:k]