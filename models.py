from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier


def modelo(treino, treino_classe, tipo):
    if tipo is 'naive':
        modelo = naive_bayes.MultinomialNB()
    elif tipo is 'logistic':
        modelo = linear_model.LogisticRegression(solver='lbfgs')
    elif tipo is 'svm':
        #modelo = svm.SVC(kernel="linear")
        modelo = svm.LinearSVC()
    elif tipo is 'random':
        modelo = RandomForestClassifier()
    elif tipo is 'tree':
        modelo = DecisionTreeClassifier()

    x = modelo.fit(treino, treino_classe)
    return x


# retorna as classes dos documentos de teste
def teste(teste, modelo):
    return modelo.predict(teste)
