
from sklearn import feature_extraction
from sklearn import feature_selection
import delta

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


def feature_selection_methods(treino, classes, teste, metodo, k, features):
    # Selecionar os k melhores ranqueados de acordo com o método
    if metodo is "chi":
        funct = feature_selection.SelectKBest(feature_selection.chi2, k=k)
    elif metodo is "anova":
        funct = feature_selection.SelectKBest(feature_selection.f_classif, k=k)
    else: return treino, teste

    treino = funct.fit_transform(treino, classes)
    teste = funct.transform(teste)

 #   cols = funct.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
  #  for i in cols:
   #     print(features[i])

    return treino, teste




