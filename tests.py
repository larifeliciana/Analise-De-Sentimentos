from sklearn import metrics
import features as f
import models as m
import nltk
import data as d
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
    tamanho = len(data) / k #400
    results = []
    for i in range(k):
        inicio = int(tamanho * i)
        fim = int(inicio + tamanho)
        test = data[inicio:fim]
        teste_labels = labels[inicio:fim]
        trein = data[0:inicio] + data[fim:len(data)]
        trein_labels = labels[0:inicio] + labels[fim:len(data)]

        trein, test = f.feature_extraction_methods(trein, test, feature_extraction1, True, False)
        trein, test = f.feature_selection_methods(trein, trein_labels, test, feature_selection1, n_features)

        modelo = m.modelo(trein, trein_labels, algoritmo)
        predito = m.teste(test, modelo)

        results.append(avaliacao(metrica, teste_labels, predito))

    print(results)
    return sum(results)/k

def cross_domain(data, labels, algoritmo, metrica, feature_extraction1, feature_selection1, n_features):#data:lista cada elemento é um dataset
    numero_datasets = len(data)
    results = []
    results1 = []
    for i in range(numero_datasets):
        print(i)
        teste = data[i]
        teste_classes = labels[i]
        treino = []
        treino_classes = []

        for i in range(0,i):
            treino = data[i]
            treino_classes = labels[i]


        for i in range(i+1, numero_datasets):
           treino = treino + data[i]
           treino_classes = treino_classes + labels[i]

        treino, teste = f.feature_extraction_methods(treino, teste, feature_extraction1, True, False)
        treino, teste = f.feature_selection_methods(treino, treino_classes, teste, feature_selection1, n_features)

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

    z = [(a,b,c,d) for a in lista1 for b in lista2 for c in lista3 for d in lista4]
    return z

def grid_search(parametros):
    p = gerar_parametros(parametros)
    dic = {}
    for i in p:
        print(datetime.datetime.now())
        a,b,c,x = i
        x,y = main(a, b, 'acuracia', c, x)

        dic.update({i:(x,y)})

    d.salvar('logistic',dic)

def main(algoritmo, feature_extraction1, metrica, feature_selection1, n_features):
    print('e')
    data = []
    labels = []
    datasets = ['books.pk','eletronics.pk', 'clothes.pk', 'cds.pk', 'movies.pk']
    for i in datasets:
        x, y = d.ler(i)
        data.append(x)
        labels.append(y)

    for i in range(len(data)):
        doc = data[i]
        classes = labels[i]
        c = list(zip(doc, classes))

        random.shuffle(c)

        a, b = zip(*c)
        data[i] = a
        labels[i] = b

    return cross_domain(data,labels,algoritmo, metrica, feature_extraction1,feature_selection1, n_features)


#parametros_grid = {'algoritmo':['logistic'], 'feature_extraction1':['tfidf'],'feature_selection1':['chi','anova'], 'n_features':[i for i in range(100, 6000, 500)]}



#grid_search(parametros_grid)
#,'idf', 'binario', 'counter'
#, 'anova'

main('tree', 'tfidf', 'acuracia', 'chi', 100)