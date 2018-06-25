import numpy as np
import os
import re
import pickle
### label mapping
label_map = [ 'Other',
        'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)' ,
        'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)' ,
        'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)' ,
        'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)' , 
        'Cause-Effect(e1,e2)' , 'Cause-Effect(e2,e1)' ,
        'Component-Whole(e1,e2)' , 'Component-Whole(e2,e1)' , 
        'Entity-Origin(e1,e2)' , 'Entity-Origin(e2,e1)' ,
        'Member-Collection(e1,e2)' , 'Member-Collection(e2,e1)' , 
        'Content-Container(e1,e2)' , 'Content-Container(e2,e1)'
        ]
datadir_path = '../dataset/'
train_path = os.path.join(datadir_path, 'TRAIN_FILE.txt')
test_path = os.path.join(datadir_path, "TEST_FILE.txt")
embedding_path = '../../deps.words'
maxsen_len = 96
def load_train_data(path):
    train_word = []
    train_Pos1= []
    train_Pos2 = []
    train_relation = []
    with open(path, 'r') as f:
        content = f.readlines()
    content = [x for x in content]
    for idx in range(0, len(content), 4):
        words = content[idx].split('\t')[1][1:-3]
        split_word = words.split()
        for word in split_word:
            if '<e1>' in word:
                pos_e1 = split_word.index(word)
                train_Pos1.append(pos_e1)
            if '<e2>' in word:
                pos_e2 = split_word.index(word)
                train_Pos2.append(pos_e2)
        words = words.replace('<e1>','').replace('</e1>','') 
        words = words.replace('<e2>','').replace('</e2>','')
        words = clear_str(words)
        train_word.append(words)
        relation = content[idx+1].strip()
        if relation in label_map:
            train_relation.append(label_map.index(relation))
    return train_word, train_Pos1, train_Pos2, train_relation 
def load_test_data(path):
    test_word = []
    test_Pos1 = []
    test_Pos2 = []
    count = 0
    with open(path, 'r') as f:
        for line in f:
            _, sentence = line.split('\t')
            words = sentence[1:-3]
            split_word = words.split()
            for word in split_word:
                if '<e1>' in word:
                    pos_e1 = split_word.index(word)
                    test_Pos1.append(pos_e1)
                if '<e2>' in word:
                    pos_e2 = split_word.index(word)
                    test_Pos2.append(pos_e2)
            words = words.replace('<e1>','').replace('</e1>','') 
            words = words.replace('<e2>','').replace('</e2>','')
            words = clear_str(words)
            test_word.append(words)
            count += 1
    return test_word, test_Pos1, test_Pos2

def clear_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
    
def create_wordset(train_word, test_word):
    wordset = set()
    for sentence in train_word:
        for word in sentence.split():
            wordset.add(word)
    return list(wordset)

def create_embedding(embedding_path,train_word, test_word):
    word2Idx = {}
    embeddings = []
    words = create_wordset(train_word, test_word)
    print('len words:',len(words))
    count = 0
    for line in open(embedding_path):
        if count == 0:
            print(line)
        split = line.strip().split(' ')
        word = split[0]
        count += 1
        if len(word2Idx) == 0:
            word2Idx['<Pad>'] = len(word2Idx)
            vector = np.zeros(len(split)-1)
            embeddings.append(vector)

            word2Idx['<UNK>'] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            embeddings.append(vector)
        if split[0] in words:
            vector = np.array([float(num) for num in split[1:]])
            embeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)
    embeddings = np.array(embeddings)
    np.save('emb.npy', embeddings)
    with open('word2IDx.pickle', 'wb') as f:
        pickle.dump(word2Idx, f)

def create_data(word2Idx, flag):
    if flag == 'train':
        words, pos1, pos2, label = load_train_data(train_path)
        label = np.array(label)
    elif flag == 'test':
        words, pos1, pos2 = load_test_data(test_path)
    tokenMatrix = []
    position1Matrix = []
    position2Matrix = []
    for _id in range(len(words)):
        tokens = np.zeros(maxsen_len)
        pos1Value = np.zeros(maxsen_len)
        pos2Value = np.zeros(maxsen_len)
        e1 = pos1[_id]
        e2 = pos2[_id]
        for word in range(min(maxsen_len, len(words[_id].split()))):
            tokens[word] = getWordIdx(words[_id].split()[word], word2Idx)
            dis1 = word - int(e1)
            dis2 = word - int(e2)
            if dis1 in list(range(-30, 30)):
                pos1Value[word] = dis1 + 32
            elif dis1 < -30:
                pos1Value[word] = 0
            else:
                pos1Value[word] = 1
            if dis2 in list(range(-30, 30)):
                pos2Value[word] = dis2 + 32
            elif dis2 < -30:
                pos2Value[word] = 0
            else:
                pos2Value[word] = 1
        tokenMatrix.append(tokens)
        position1Matrix.append(pos1Value)
        position2Matrix.append(pos2Value)
    if flag == 'train':
        return np.array(tokenMatrix, dtype='int32'), np.array(position1Matrix, dtype='int32'), np.array(position2Matrix, dtype='int32'), label
    elif flag == 'test':
        return np.array(tokenMatrix, dtype='int32'), np.array(position1Matrix, dtype='int32'), np.array(position2Matrix, dtype='int32')
    

def getWordIdx(word, word2Idx):
    if word in word2Idx:
        return word2Idx[word]
    return word2Idx['<UNK>']


if __name__ == '__main__':
    test_word, test_pos1, test_pos2 = load_test_data(test_path)
    with open('word2IDx.pickle','rb') as f:
        word2Idx = pickle.load(f)
    train_token, train_pos1, train_pos2, train_y= create_data(word2Idx, 'train')
    test_token, test_pos1, test_pos2 = create_data(word2Idx, 'test')
