from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

def modelo(treino, treino_classe, tipo):
    if tipo is 'naive': #Naive Bayes
        modelo = naive_bayes.MultinomialNB()
    elif tipo is 'logistic':  # Regressão Logística
        modelo = linear_model.LogisticRegression()
    elif tipo is 'svm':
        modelo = svm.SVC(kernel="linear")
    elif tipo is 'random':
        modelo = RandomForestClassifier()
    elif tipo is 'tree':
        modelo = DecisionTreeClassifier()



    x = modelo.fit(treino,treino_classe )
    return x

def teste(teste, modelo): #retorna as classes dos documentos de teste
    return modelo.predict(teste)

