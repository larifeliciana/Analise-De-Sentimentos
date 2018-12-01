from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import features as f
import models as m
import data as d
from itertools import permutations

def avaliacao(metrica, classes, teste):
    if metrica is 'acuracia':
        return metrics.accuracy_score(y_true=classes, y_pred=teste)
    if metrica is 'f-mesure':
        return metrics.f1_score(y_true=classes, y_pred=teste)

#def cross_validation():

#def cross_domain():

def gerar_parametros(parametros):
    import itertools
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
        a,b,c,x = i
        x=main(a,b,'acuracia', False, c,x)
        dic.update({i:x})

    d.salvar('results',dic)

def main(algoritmo, feature_extraction1, metrica, smooth, feature_selection1,  n_features):

    treino1, treino_classes, teste1, teste_classes = d.ler('movies')

    treino1, teste1 = f.feature_extraction_methods(treino1, teste1, feature_extraction1, True, smooth)

    treino1, teste1 = f.feature_selection_methods(treino1, treino_classes, teste1, feature_selection1 ,n_features)

    modelo1 = m.modelo(treino1,treino_classes,algoritmo)

    predito = m.teste(teste1,modelo1)

    return avaliacao(metrica, teste_classes,predito)

#main("logistic", "idf", "acuracia", True, "chi", 5000)

parametros_grid = {'algoritmo':['logistic', 'naive'], 'feature_extraction1':['tfidf', 'binario', 'counter'],'feature_selection1':['chi', 'anova'], 'n_features':[i for i in range(100, 6000, 500)]}

grid_search(parametros_grid)
#clf = GridSearchCV(main, parametros_grid, cv=5)
#print(clf.fit())