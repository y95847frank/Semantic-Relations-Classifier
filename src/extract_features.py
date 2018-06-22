import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
                    y[j] = 1
                else:
                    y[j+10] = 1
        train_x.append(x)
        train_y.append(y)
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
    np.save('train_x', np.array(train_x))
    np.save('train_y', np.array(train_y))
    np.save('test_x' , np.array(test_x))

def generate_result():
    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')
    test_x = np.load('test_x.npy')
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(train_x.shape[1],)))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(train_x, train_y, batch_size=32)
    test_y = model.predict(test_x)
    output = open('myresult.txt', 'w')
    count = 8001
    for y in test_y:
        y = np.argmax(y)
        if y == 19:
            output.write(str(count) + '\t' + relation_type[y-10] + '\r\n')
        if y >= 10:
            output.write(str(count) + '\t' + relation_type[y-10] + '(e2,e1)\r\n')
        else:
            output.write(str(count) + '\t' + relation_type[y] + '(e1,e2)\r\n')
        count += 1
    output.close()

generate_result()       
