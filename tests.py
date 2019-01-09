from sklearn import metrics
import features as f
import models as m
import nltk
import data as dt
import math
import datetime
import random
from itertools import permutations


def avaliacao(metrica, classes, teste):
    if metrica is 'acuracia':
        return metrics.accuracy_score(y_true=classes, y_pred=teste)
    if metrica is 'fmesure':
        return metrics.f1_score(y_true=classes, y_pred=teste)


def cross_validation(data, labels, k, algoritmo, metrica, feature_extraction1, feature_selection1, n_features):
    tamanho = len(data) / k     # 400
    results = []
    for i in range(k):
        print(i)
        inicio = int(tamanho * i)
        fim = int(inicio + tamanho)
        test1= data[inicio:fim]
        teste_labels = labels[inicio:fim]
        trein1 = data[0:inicio] + data[fim:len(data)]
        trein_labels = labels[0:inicio] + labels[fim:len(data)]

        equal = None
        trein, test, pre = f.feature_extraction_methods(trein1, test1, feature_extraction1, True, False, trein_labels)
        if feature_selection1 is feature_extraction1:
            equal = pre, trein

        trein, test = f.feature_selection_methods(trein, trein_labels, test, feature_selection1, n_features, trein1, test1, equal)

        modelo = m.modelo(trein, trein_labels, algoritmo)
        predito = m.teste(test, modelo)

        results.append(avaliacao(metrica, teste_labels, predito))

    print(results)
    return sum(results)/k


def cross_domain(data, labels, algoritmo, feature_extraction1, feature_selection1, n_features):#data:lista cada elemento é um dataset
    numero_datasets = len(data)
    results = []
    results1 = []
    for j in range(numero_datasets):
        print(j)
        test1 = data[j]
        teste_classes = labels[j]
        trein1 = []
        treino_classes = []

        for i in range(j):
            trein1 = data[i]
            treino_classes = labels[i]

        for i in range(j+1, numero_datasets):
           trein1 = trein1 + data[i]
           treino_classes = treino_classes + labels[i]


        treino, teste, pre = f.feature_extraction_methods(trein1, test1, feature_extraction1, True, False, treino_classes)
        treino, teste = f.feature_selection_methods(treino, treino_classes, teste, feature_selection1,
                                                    n_features)

        modelo = m.modelo(treino, treino_classes, algoritmo)
        predito = m.teste(teste, modelo)
        results.append(avaliacao('acuracia', teste_classes, predito))
        results1.append(avaliacao('fmesure', teste_classes, predito))

    print(results)
    return sum(results) / numero_datasets, sum(results1) / numero_datasets


def checar(lista, n):
    lista1 = []
    for i in lista:
        if len(nltk.sent_tokenize(i)) > n:
            lista1.append(i)
    return lista1


def gerar_parametros(parametros):

    lista1 = parametros['algoritmo']
    lista2 = parametros['feature_extraction1']
    lista3 = parametros['feature_selection1']
    lista4 = parametros["n_features"]

    z = [(a, b, c, d) for a in lista1 for b in lista2 for c in lista3 for d in lista4]
    return z


def grid_search(parametros):
    p = gerar_parametros(parametros)
    dic = {}
    for i in p:
        print(datetime.datetime.now())

        a, b, c, x = i
        print(i)
        x, y = main(a, b, c, x)

        dic.update({i: (x, y)})

        dt.salvar('logistic_1', dic)


def main(algoritmo, feature_extraction1, feature_selection1, n_features):
    data = []
    labels = []
    datasets = ['books.pk', 'electronics.pk', 'clothes.pk', 'cds.pk', 'movies.pk']
    for i in datasets:
        x, y = dt.ler(i)
        data.append(x)
        labels.append(y)

    #data, labels = dt.ler("books.pk")
    return cross_domain(data, labels ,algoritmo, feature_extraction1, feature_selection1,n_features)


parametros_grid = {'algoritmo': ['logistic'],
                   'feature_extraction1': ['tfidf','idf', 'counter', 'binario', 'delta'],
                   'feature_selection1': ['chi', 'anova','fvalue'],
                   'n_features': [i for i in range(100, 4101, 500)]}


def sort_dict(dic):
    import operator
    return sorted(dic.items(), key=operator.itemgetter(1), reverse=True)

grid_search(parametros_grid)
#print(dt.ler('grid1'))