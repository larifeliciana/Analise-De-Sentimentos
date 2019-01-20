from sklearn import metrics
import features as f
import models as m
import nltk
import data as dt
import math
import datetime
import random
import os
from itertools import permutations
#import SpectralAlignment.SFA


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


def cross_domain(domain, algoritmo, feature_extraction1, feature_selection1, n_features, time):#data:lista cada elemento é um dataset

    datasets = ['books', 'kitchen', 'dvd','electronics']
    src = datasets[domain[0]]
    dst = datasets[domain[1]]
    trein1, treino_classes = dt.ler('Data1/' + src + '.pk')
    test1, teste_classes = dt.ler('Data1/' + dst + '.pk')

    treino, teste, pre = f.feature_extraction_methods(trein1, test1, feature_extraction1, True, False, treino_classes)
    treino, teste = f.feature_selection_methods(treino, treino_classes, teste, feature_selection1,
                                                n_features, pre.get_feature_names(), trein1, test1)


    modelo = m.modelo(treino, treino_classes, algoritmo)
    predito = m.teste(teste, modelo)
    results  = avaliacao('acuracia', teste_classes, predito)
    time = (datetime.datetime.now() - time).total_seconds()

    sentence = "Accuracy = " + str(results) + "Time = " + str(time)

    filename = 'Results/'+algoritmo+"/"+src+"_to_"+dst+"_"+feature_extraction1+"_"+feature_selection1
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    print(filename)
    print(sentence)
    with open(filename, "w") as myfile:
        myfile.write(sentence+"\n")

    return results

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
    lista5 = parametros["domain"]
    z = [(e, a, b, c, d) for a in lista1 for b in lista2 for c in lista3 for d in lista4 for e in lista5]
    return z


def grid_search(parametros):
    p = gerar_parametros(parametros)
    dic = {}
    for i in p:
        print(datetime.datetime.now())

        a, b, c, x, z = i
        print(i)

        x = cross_domain(a, b, c, x, z, datetime.datetime.now())

#        dic.update({i: (x, y)})

#        dt.salvar('logistic_1', dic)



parametros_grid = {'algoritmo': ['logistic','svm','random', 'tree'],
                   'feature_extraction1': ['tfidf','idf', 'counter', 'binario'],
                   'feature_selection1': ['chi', 'anova','tfidf','delta'],
                   'domain':[[0,1],[0,2],[0,3],[1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]],
                   'n_features': [1100]}


def sort_dict(dic):
    import operator
    return sorted(dic.items(), key=operator.itemgetter(1), reverse=True)

grid_search(parametros_grid)

