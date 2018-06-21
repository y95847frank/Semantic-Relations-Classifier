import numpy as np
import pickle
from sklearn import svm

relation_type = ['Cause-Effect', 'Instrument-Agency', 'Product-Producer', 'Content-Container', 'Entity-Origin', 'Entity-Destination', 'Component-Whole', 'Member-Collection', 'Message-Topic', 'Other']
def lexical():
    with open('dataset/TRAIN_FILE.txt') as f:
        content = f.readlines()
    content = [ x for x in content ]
    word_set = set()
    for i in range(0, len(content), 4):
        words = content[i].split('\t')[1][1:-2].split()
        for word in words:
            if '<e1>' in word or '<e2>' in word:
                word = word[5:-5]
            word_set.add(word)
    word_set = list(word_set)
    word2index = { word_set[i]:i for i in range(len(word_set)) }
    train_x = []
    train_y = []
    for i in range(0, len(content), 4):
        words = content[i].split('\t')[1][1:-1].split()
        x = np.zeros(len(word_set))
        y = np.zeros(2 * len(relation_type))
        for word in words:
            if '<e1>' in word or '<e2>' in word:
                word = word[5:-5]
            x[word2index[word]] = 1
        for j in range(len(relation_type)):
            if relation_type[j] in content[i+1]:
                if '(e1' in content[i+1]:
                    train_y.append(j)
                else:
                    train_y.append(j+10)
        train_x.append(x)
    np.save('train_x', np.array(train_x))
    np.save('train_y', np.array(train_y))
    return word2index

def generate_result(word2index):
    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')
    clf = svm.SVC()
    clf.fit(train_x, train_y)
    with open('dataset/TEST_FILE.txt', 'r') as f:
        content = f.readlines()
    test_x = []
    for line in content:
        _, sentence = line.split('\t')
        x = np.zeros(len(word2index))
        for word in sentence[1:-2]:
            if word in word2index:
                x[word2index[word]] = 1
        test_x.append(x)
    test_y = clf.predict(np.array(test_x))
    output = open('myresult.txt', 'w')
    count = 8001
    for y in test_y:
        if y >= 10:
            output.write(str(count) + '\t' + relation_type[y-10] + '(e2, e1)\n')
        else:
            output.write(str(count) + '\t' + relation_type[y] + '(e1, e2)\n')
w2i = lexical()
generate_result(w2i)
