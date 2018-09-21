import pickle
from sklearn import feature_extraction
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import metrics

def ler(endereco):
    arq = open(endereco, 'rb')
    return pickle.load(arq)


def salvar(endereco, lista):
    arq = open(endereco, 'wb')
    pickle.dump(lista, arq)

def preprocessamento(treino, teste, tipo, stopwords, max_features):
#dataset, o tipo da representação, se possui pontuações, a máxima frequência, a mínima frequência, o maximo de features
    if stopwords:
        stopwords='english'

    if tipo is 'tfidf':
        pre = feature_extraction.text.TfidfVectorizer(stop_words=stopwords,max_features=max_features)
    elif tipo is 'tf':
        pre = feature_extraction.text.CountVectorizer(stop_words=stopwords,max_features=max_features)
    elif tipo is 'binario':
        pre = feature_extraction.text.CountVectorizer(stop_words=stopwords,max_features=max_features, binary=True)

    return pre.fit_transform(treino), pre.transform(teste)


def modelo(treino, treino_classe, tipo):
    if tipo is 'naive': #Naive Bayes
        modelo = naive_bayes.MultinomialNB()
    elif tipo is 'logistic':  # Regressão Logística
        modelo = linear_model.LogisticRegression()

    x = modelo.fit(treino,treino_classe )
    return x

def teste(teste, modelo): #retorna as classes dos documentos de teste
    return modelo.predict(teste)


def avaliacao(metrica, classes, teste):
    if metrica is 'acuracia':
        return metrics.accuracy_score(y_true=classes, y_pred=teste)


treino1, treino_classes, teste1, teste_classes = ler('movies')

treino1, teste1 = preprocessamento(treino1, teste1, 'tfidf', True, 3000)

modelo1 = modelo(treino1,treino_classes, 'logistic')
predito = teste(teste1,modelo1)

print(avaliacao('acuracia', teste_classes,predito))
