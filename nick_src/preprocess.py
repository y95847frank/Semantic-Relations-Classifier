import numpy as np
import os
import re
import pickle
import nltk
from nltk.corpus import wordnet as wn
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
    count = 0
    for idx in range(0, len(content), 4): #len(content)
        words = content[idx].split('\t')[1][1:-3]
        words = words.replace('<e1>', '_e1').replace('</e1>','e1')
        words = words.replace('<e2>', '_e2').replace('</e2>','e2')
        words = clear_str(words)
        split_word = words.split()
        pos1_flag = False
        pos2_flag = False
        for word in split_word:
            if 'e1' in word and pos1_flag == False:
                pos_e1 = split_word.index(word)
                train_Pos1.append(pos_e1)
                pos1_flag = True
            if 'e2' in word and pos2_flag == False:
                pos_e2 = split_word.index(word)
                train_Pos2.append(pos_e2)
                pos2_flag = True
        if count == 2134:
            print(words)
            print(pos_e1, pos_e2)
        count += 1
        words = words.replace('e1','')
        words = words.replace('e2','')
        train_word.append(words)
        relation = content[idx+1].strip()
        if relation in label_map:
            train_relation.append(label_map.index(relation))
    return train_word, train_Pos1, train_Pos2, train_relation 
def load_test_data(path):
    test_word = []
    test_Pos1 = []
    test_Pos2 = []
    with open(path, 'r') as f:
        for line in f:
            _, sentence = line.split('\t')
            words = sentence[1:-3]
            words = words.replace('<e1>', '_e1').replace('</e1>','e1')
            words = words.replace('<e2>', '_e2').replace('</e2>','e2')
            words = clear_str(words)
            split_word = words.split()
            pos1_flag = False
            pos2_flag = False
            for word in split_word:
                if 'e1' in word and pos1_flag == False:
                    pos_e1 = split_word.index(word)
                    test_Pos1.append(pos_e1)
                    pos1_flag = True
                if 'e2' in word and pos2_flag == False:
                    pos_e2 = split_word.index(word)
                    test_Pos2.append(pos_e2)
                    pos2_flag = True
            words = words.replace('e1','')
            words = words.replace('e2','')
            words = clear_str(words)
            test_word.append(words)
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
    
def create_wordset(train_word):
    wordset = set()
    for sentence in train_word:
        for word in sentence.split():
            wordset.add(word)
            try:
                wordset.add(wn.synsets(word)[0].hypernyms()[0].name().split('.')[0])
            except:
                a = 0
    return list(wordset)

def create_pos_set(train_word, train_pos1, train_pos2):
    pos1_list = list()
    pos2_list = list()
    count = 0
    for sentence in train_word:
        pos_position1 = nltk.pos_tag(sentence.split())[train_pos1[count]][1] 
        pos_position2 = nltk.pos_tag(sentence.split())[train_pos2[count]][1]

        if pos_position1 not in pos1_list:
            pos1_list.append(pos_position1)
        if pos_position2 not in pos2_list:
            pos2_list.append(pos_position2)
        count += 1
    print(pos1_list)
    print(pos2_list)

    with open('pos1_list.pkl','wb') as f:
        pickle.dump(pos1_list, f)
    with open('pos2_list.pkl','wb') as f:
        pickle.dump(pos2_list, f)
    
    #return pos1_list, pos2_list

def create_embedding(embedding_path,train_word, test_word):
    word2Idx = {}
    embeddings = []
    words = create_wordset(train_word)
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
def get_pos_index(pos_list, word):
    word = list(word)
    pos = nltk.pos_tag(word)[0][1]
    if pos not in pos_list:
        return 0
    else:
        return pos_list.index(pos) + 1

def create_data(word2Idx, flag, pos_pos1_list, pos_pos2_list):
    if flag == 'train':
        words, pos1, pos2, label = load_train_data(train_path)
        label = np.array(label)
    elif flag == 'test':
        words, pos1, pos2 = load_test_data(test_path)
    tokenMatrix = []
    position1Matrix = []
    position2Matrix = []
    position1_5Word = [] 
    pos_position1Matrix = []
    pos_position2Matrix = []
    
    for _id in range(len(words)):
        wordvec = []
        tokens = np.zeros(maxsen_len)
        pos1Value = np.zeros(maxsen_len)
        pos2Value = np.zeros(maxsen_len)
        e1 = pos1[_id]
        e2 = pos2[_id]

        
        got_index = getWordIdx(words[_id].split()[e1], word2Idx)
        wordvec.append(got_index)
        got_index = getWordIdx(words[_id].split()[e2], word2Idx)
        wordvec.append(got_index)
        
        tmp_e1 = words[_id].split()[e1]
        tmp_e2 = words[_id].split()[e2]
        try:
            wordvec.append(getWordIdx(wn.synsets(tmp_e1)[0].hypernyms()[0].name().split('.')[0], word2Idx))
        except:
            wordvec.append(e1)
        try:
            wordvec.append(getWordIdx(wn.synsets(tmp_e2)[0].hypernyms()[0].name().split('.')[0], word2Idx))
        except:
            wordvec.append(e2)
        
        if e1 == 0:
            got_index = getWordIdx(words[_id].split()[e1], word2Idx)
            wordvec.append(got_index)
        else:
            got_index = getWordIdx(words[_id].split()[e1-1], word2Idx)
            wordvec.append(got_index)
        
        got_index = getWordIdx(words[_id].split()[e1+1], word2Idx)
        wordvec.append(got_index)
        got_index = getWordIdx(words[_id].split()[e2-1], word2Idx)
        wordvec.append(got_index)
        if e2 == len(words[_id].split()):
            got_index = getWordIdx(words[_id].split()[e2], word2Idx)
            wordvec.append(got_index)
        else:
            got_index = getWordIdx(words[_id].split()[e2-1], word2Idx)
            wordvec.append(got_index)

        pos1_idx = get_pos_index(pos_pos1_list, words[_id].split()[e1])
        pos2_idx = get_pos_index(pos_pos2_list, words[_id].split()[e2])
        pos_position1Matrix.append(pos1_idx)
        pos_position2Matrix.append(pos2_idx)
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
        position1_5Word.append(wordvec)
    #print(np.array(position1_5Word).shape)
    #print(np.array(pos_position1Matrix).shape)
    #print(np.array(pos_position2Matrix).shape)
    if flag == 'train':
        return np.array(tokenMatrix, dtype='int32'), np.array(position1Matrix, dtype='int32'), np.array(position2Matrix, dtype='int32'), np.array(position1_5Word, dtype='int32'), label
    elif flag == 'test':
        return np.array(tokenMatrix, dtype='int32'), np.array(position1Matrix, dtype='int32'), np.array(position2Matrix, dtype='int32'), np.array(position1_5Word, dtype='int32')
    

def getWordIdx(word, word2Idx):
    if word in word2Idx:
        return word2Idx[word]
    return word2Idx['<UNK>']


if __name__ == '__main__':
    #test_word, test_pos1, test_pos2 = load_test_data(test_path)
    with open('word2IDx.pickle','rb') as f:
        word2Idx = pickle.load(f)
    '''
    train_token, train_pos1, train_pos2, train_y= create_data(word2Idx, 'train')
    test_token, test_pos1, test_pos2 = create_data(word2Idx, 'test')
    '''
    train_word, pos1, pos2, label = load_train_data(train_path)
    create_pos_set(train_word, pos1, pos2)
    #pos_pos1_list, pos_pos2_list = create_pos_set(train_word, pos1, pos2)
    '''
    with open('pos1_list.pkl','rb') as f:
        pos_pos1_list = pickle.load(f)
    with open('pos2_list.pkl','rb') as f:
        pos_pos2_list = pickle.load(f)
    '''
    #print(len(pos_pos1_list))
    #print(len(pos_pos2_list))
    #train_token, train_pos1, train_pos2, train_POS1, train_POS2, train_y= create_data(word2Idx, 'train', pos_pos1_list, pos_pos2_list)
    #test_token, test_pos1, test_pos2, test_POS1, test_POS2 = create_data(word2Idx, 'test', pos_pos1_list, pos_pos2_list)
