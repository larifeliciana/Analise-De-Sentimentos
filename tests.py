from sklearn import metrics
import features as f
import models as m
import data as d
def avaliacao(metrica, classes, teste):
    if metrica is 'acuracia':
        return metrics.accuracy_score(y_true=classes, y_pred=teste)
    if metrica is 'f-mesure':
        return metrics.f1_score(y_true=classes, y_pred=teste)

#def cross_validation():

#def cross_domain():

#def grid_search():

def main(algoritmo, feature_extraction1, metrica, smooth, feature_selection1,  n_features):

    treino1, treino_classes, teste1, teste_classes = d.ler('movies')

    treino1, teste1 = f.feature_extraction_methods(treino1, teste1, feature_extraction1, True, smooth)

    treino1, teste1 = f.feature_selection_methods(treino1, treino_classes, teste1, feature_selection1 ,n_features)

    modelo1 = m.modelo(treino1,treino_classes,algoritmo)

    predito = m.teste(teste1,modelo1)

    print(avaliacao(metrica, teste_classes,predito))

main("logistic", "idf", "acuracia", True, "chi", 5000)

#parametros_grid = {'algoritmo':['logistic', 'naive'], 'feature_extraction1':['tfidf', 'binario', 'counter'],'feature_selection1':['chi', 'anova'] 'n_features':[i for i in range(100, 6000, 500)]}