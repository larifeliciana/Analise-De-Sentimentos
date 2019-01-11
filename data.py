import pickle
import os


def ler(endereco):
    arq = open(endereco, 'rb')
    return pickle.load(arq)


def carregar(pasta):
    caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
    lista = []
    for i in caminhos:
        review = open(i, 'r',  encoding="utf8")
        lista.append(review.read())
    return lista


def salvar(endereco, lista):
    arq = open(endereco, 'wb')
    pickle.dump(lista, arq)


def dataset(nome):
    x = ler(nome)
    return x[0],x[1]
