import numpy as np
import pickle
import argparse
from CNN import CNN
from preprocess import create_data
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight

max_sentence_length = 96
position_dim =25
num_epoch = 100
batch = 100
n_out = 19
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
args = parser.parse_args()

with open('pos1_list.pkl','rb') as f:
    pos_pos1_list = pickle.load(f)
with open('pos2_list.pkl','rb') as f:
    pos_pos2_list = pickle.load(f)

label_map = ['Other',
                'Message-Topic(e1,e2)' , 'Message-Topic(e2,e1)', 
                'Product-Producer(e1,e2)' , 'Product-Producer(e2,e1)', 
                'Instrument-Agency(e1,e2)' , 'Instrument-Agency(e2,e1)', 
                'Entity-Destination(e1,e2)' , 'Entity-Destination(e2,e1)',
                'Cause-Effect(e1,e2)' , 'Cause-Effect(e2,e1)',
                'Component-Whole(e1,e2)' , 'Component-Whole(e2,e1)',  
                'Entity-Origin(e1,e2)' , 'Entity-Origin(e2,e1)',
                'Member-Collection(e1,e2)' , 'Member-Collection(e2,e1)',
                'Content-Container(e1,e2)' , 'Content-Container(e2,e1)'
                ]
embedding = np.load('emb.npy')
print(embedding.shape)

modOpts = {
        'maxsen_len' : max_sentence_length,
        'max_pos' : 62,
        'pos_dim' : position_dim,
        'vocab_size' : embedding.shape[0],
        'emb_dim' : embedding.shape[1],
        'filter_len_list' : [1],
        'num_filter' : 500,
        'n_class' : 19,
        'embedding' : embedding,
        'POS_size' : 15
        }


model = CNN(modOpts).build_model()
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

### split training data to (train, validation)
def split_data(token, pos1, pos2, Y, lexical, split_ratio):
    
    index = np.arange(token.shape[0])
    token_data = token[index]
    y_data = Y[index]
    lexical_data = lexical[index]
    pos1_data = pos1[index]
    pos2_data = pos2[index]

    bound = int(token.shape[0] * split_ratio)
    (train_lexical, val_lexical) = (lexical_data[bound:], lexical_data[:bound])
    (train_token, val_token) = (token_data[bound:], token_data[:bound])
    (train_pos1, val_pos1) = (pos1_data[bound:], pos1_data[:bound])
    (train_pos2, val_pos2) = (pos2_data[bound:], pos2_data[:bound])
    (train_y, val_y) = (y_data[bound:], y_data[:bound])
    
    return (train_token, val_token), (train_pos1, val_pos1), (train_pos2, val_pos2), (train_lexical, val_lexical), (train_y, val_y)

with open('word2IDx.pickle', 'rb') as f:
   
    word2IDx = pickle.load(f)

    train_token, train_pos1, train_pos2, train_lexical, train_y = create_data(word2IDx, 'train', pos_pos1_list, pos_pos2_list)
    count = 0
    for i in range(train_y.shape[0]):
        if train_y[i] == 0:
            count += 1
    print('count:', count)
    #class_weight_dict = {}
    #print(class_weight_dict)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
    print(class_weights)
    class_weight_dict = dict(enumerate(class_weights))
    print(class_weight_dict)

    train_y = np_utils.to_categorical(train_y, n_out)

    test_token, test_pos1, test_pos2, test_lexical = create_data(word2IDx, 'test', pos_pos1_list, pos_pos2_list)
    (train_token, val_token), (train_pos1, val_pos1), (train_pos2, val_pos2), (train_lexical, val_lexical), (train_y, val_y) = split_data(train_token, train_pos1, train_pos2, train_y, train_lexical,  0.2)
    
    if args.train:
        filepath = "weights_CNN_best_unstable_wordemb_POS1_POS2.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose = 1, save_best_only=True, mode='min')
        earlystop = EarlyStopping(monitor='val_loss', patience=2, mode='min')
        callbacks_list = [checkpoint, earlystop]
        model.fit([train_pos1, train_pos2, train_token, train_lexical], train_y, epochs=num_epoch, batch_size=batch, validation_data = ([val_pos1, val_pos2, val_token, val_lexical], val_y), class_weight=class_weight_dict,callbacks=callbacks_list, verbose=1)
    else:
        #model.load_weights('best.h5')
        model.load_weights('weights_CNN_best_unstable_wordemb_POS1_POS2.h5')
        predict = model.predict([test_pos1, test_pos2, test_token, test_lexical])
        predict_cls = np.argmax(predict, axis=1)
        print(predict_cls.shape)
        print(predict_cls[208])
        print(label_map[17])
        count = 8001
        with open('../dataset/myans.txt','w') as f:
            for item in range(predict_cls.shape[0]): #17018
                #print(item)
                f.write(str(count) + '\t' + label_map[predict_cls[item]] + '\n')
                count += 1

