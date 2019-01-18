
from sklearn import feature_extraction
from sklearn import feature_selection
import delta
import numpy as np


def feature_extraction_methods(treino, teste, tipo, stopwords, smooth, labels):
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
        pre = delta.DeltaTfidfVectorizer(stop_words=stopwords)

    return pre.fit_transform(treino,labels), pre.transform(teste), pre#, pre.get_feature_names()


def feature_selection_methods(treino, classes, teste, metodo, k, feature_names): ##precisa dos dados originais para selectionar com o tfidf e o delta
    stopwords = 'english'

    # Selecionar os k melhores ranqueados de acordo com o método
    if metodo is "chi":
        funct = feature_selection.SelectKBest(feature_selection.chi2, k=k)
    elif metodo is "anova":
        funct = feature_selection.SelectKBest(feature_selection.f_classif, k=k)
    elif metodo is "fvalue":
        funct = feature_selection.SelectKBest(feature_selection.f_regression, k=k)

    treino = funct.fit_transform(treino, classes)
    teste = funct.transform(teste)
    mask = funct.get_support()  # list of booleans
    new_features = []  # The list of your K best features

    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)
    print(new_features)
    return treino, teste




