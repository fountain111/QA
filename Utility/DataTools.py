import numpy as np
import random

train_file = 'datasets/train'

test_file = 'datasets/test1'

def build_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open(train_file):
        items = line.strip().split(' ')
        for i in range(2, 3):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    for line in open(test_file):
        items = line.strip().split(' ')
        for i in range(2, 3):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab


def read_alist():
    alist = []
    for line in open(train_file):
        items = line.strip().split(' ')
        alist.append(items[3])
    return alist


def read_raw():
    raw = []
    for line in open(train_file):
        items = line.strip().split(' ')
        if items[0] == '1':
            raw.append(items)
    return raw


def next_batch(vocab, alist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, size):
        items = raw[random.randint(0, len(raw) - 1)]
        nega = rand_qa(alist)
        x_train_1.append(encode_sent(vocab, items[2], 100))
        x_train_2.append(encode_sent(vocab, items[3], 100))

        x_train_3.append(encode_sent(vocab, nega, 100))

    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)


def rand_qa(qalist):
    index = random.randint(0, len(qalist) - 1)
    return qalist[index]


def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x