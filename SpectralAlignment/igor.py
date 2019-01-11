import pickle
import random

def salvar(endereco, lista):
    arq = open(endereco, 'wb')
    pickle.dump(lista, arq)


def ler(endereco):
    arq = open(endereco, 'rb')
    return pickle.load(arq)


def dataset(nome):
    x = ler(nome)
    return x[0], x[1]


def main(domain):
    op, lb = dataset(domain + '.pk')
    unlabeled = random.sample(op, int(len(op)*0.8*0.5))
    pos = []
    neg = []
    for i in range(0, len(op)-1):
        if op[i] not in unlabeled:
            if lb[i] == 1:
                pos.append(op[i])
            else:
                neg.append(op[i])
    t = int(len(pos)*0.8*0.5)
    salvar(domain + "/" + domain + '-train-unlabeled.pk', unlabeled)
    salvar(domain + "/" + domain + '-train-positive.pk', pos[:t])
    salvar(domain + "/" + domain + '-train-negative.pk', neg[:t])
    salvar(domain + "/" + domain + '-test-positive.pk', pos[t:])
    salvar(domain + "/" + domain + '-test-negative.pk', neg[t:])
    return pos, neg


main('cds')
main('clothes')
