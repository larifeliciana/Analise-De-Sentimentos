import pickle

def ler(endereco):
    arq = open(endereco, 'rb')
    return pickle.load(arq)


def salvar(endereco, lista):
    arq = open(endereco, 'wb')
    pickle.dump(lista, arq)
