import pickle
from sklearn import feature_extraction
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import metrics
from sklearn import feature_selection

def ler(endereco):
    arq = open(endereco, 'rb')
    return pickle.load(arq)


def salvar(endereco, lista):
    arq = open(endereco, 'wb')
    pickle.dump(lista, arq)

def feature_extraction_methods(treino, teste, tipo, stopwords, smooth):
    if stopwords:
        stopwords='english'

    if tipo is 'tfidf':
        pre = feature_extraction.text.TfidfVectorizer(stop_words=stopwords,smooth_idf = smooth)
    elif tipo is 'idf':
        pre = feature_extraction.text.TfidfVectorizer(stop_words=stopwords, binary=True)
    elif tipo is 'counter':
        pre = feature_extraction.text.CountVectorizer(stop_words=stopwords)
    elif tipo is 'binario':
        pre = feature_extraction.text.CountVectorizer(stop_words=stopwords)
    else: pre = feature_extraction.text.BaseEstimator

    return pre.fit_transform(treino), pre.transform(teste)

def feature_selection_methods(treino, classes, teste, metodo, k):
    if metodo is "chi":
        funct = feature_selection.SelectKBest(feature_selection.chi2, k=k)
    elif metodo is "anova":
        funct = feature_selection.SelectKBest(feature_selection.f_classif, k=k)
    else: return treino, teste

    treino = funct.fit_transform(treino, classes)
    teste = funct.transform(teste)

    return treino, teste


def modelo(treino, treino_classe, tipo):
    if tipo is 'naive': #Naive Bayes
        modelo = naive_bayes.MultinomialNB()
    elif tipo is 'logistic':  # Regressão Logística
        modelo = linear_model.LogisticRegression()
#    elif tipo is

    x = modelo.fit(treino,treino_classe )
    return x

def teste(teste, modelo): #retorna as classes dos documentos de teste
    return modelo.predict(teste)


def avaliacao(metrica, classes, teste):
    if metrica is 'acuracia':
        return metrics.accuracy_score(y_true=classes, y_pred=teste)
    if metrica is 'f-mesure':
        return metrics.f1_score(y_true=classes, y_pred=teste)



def main(algoritmo, feature_extraction1, metrica, smooth, feature_selection1,  n_features):

    treino1, treino_classes, teste1, teste_classes = ler('movies')

    treino1, teste1 = feature_extraction_methods(treino1, teste1, feature_extraction1, True, smooth)

    treino1, teste1 = feature_selection_methods(treino1, treino_classes, teste1, feature_selection1 ,n_features)

    modelo1 = modelo(treino1,treino_classes,algoritmo)

    predito = teste(teste1,modelo1)

    print(avaliacao(metrica, teste_classes,predito))

main("logistic", "idf", "acuracia", True, "chi", 5000)

#parametros_grid = {'algoritmo':['logistic', 'naive'], 'feature_extraction1':['tfidf', 'binario', 'counter'],'feature_selection1':['chi', 'anova'] 'n_features':[i for i in range(100, 6000, 500)]}