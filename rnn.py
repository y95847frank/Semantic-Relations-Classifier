import numpy as np
import data
import os
import pickle
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding , Dense ,Input , GlobalMaxPooling1D , Bidirectional , LSTM, GRU, Concatenate
from keras.models import Model
#import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # get the current directory
GLOVE_DIR = "/Users/angelocsc/Desktop/NTU/NLPLab/glove.6B"
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 25

def get_embeddinglayer(word_index):
	embeddings_index = {}
	f = open(os.path.join(GLOVE_DIR, 'glove.6B.%sd.txt' %(EMBEDDING_DIM)))
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector
	    elif(word in ['<e1>' , '</e1>' , '<e2>' , '</e2>']):
	    	if(word == '<e1>'):
	    		embedding_matrix[i] = [0.1] * (EMBEDDING_DIM-1) + [1]
	    	elif(word == '</e1>'):
	    		embedding_matrix[i] = [0.1] * (EMBEDDING_DIM-2) + [1,0.1]
	    	elif(word == '<e2>'):
	    		embedding_matrix[i] = [0.1] * (EMBEDDING_DIM-3) + [1,0.1,0.1]
	    	elif(word == '</e2>'):
	    		embedding_matrix[i] = [0.1] * (EMBEDDING_DIM-4) + [1,0.1,0.1,0.1]
	    	elif(word == 'entity1'):
	    		embedding_matrix[i] = [0.1] * (EMBEDDING_DIM-5) + [1,0.1,0.1,0.1,0.1]
	    	elif(word == 'entity2'):
	    		embedding_matrix[i] = [0.1] * (EMBEDDING_DIM-6) + [1,0.1,0.1,0.1,0.1,0.1]

	embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM,weights=[embedding_matrix],\
								input_length=MAX_SEQUENCE_LENGTH, trainable=True) # set the trainable to true, we want to train <e1>...
	return embedding_layer

def create_rnn_model(embedding_layer):
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype = 'int32')
	embedded_sequences = embedding_layer(sequence_input)

	x = Bidirectional((GRU(200 , return_sequences = True)) , merge_mode = 'concat')(embedded_sequences)
	#x = Dropout(0.5)(x)
	x = GlobalMaxPooling1D()(x)
	preds = Dense( 20 , activation='softmax')(x)

	model = Model(sequence_input , preds)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['acc'])
	return model 


def one_hot_ans(prediction):
	answer = []
	with open('tokenizer_ans.pickle' , 'rb') as handle:
			pickle.load( handle)
	word_index = tokenizer.word_index
	for predic in prediction:
		answer.append(list(word_index.keys())[list(word_index.values().index(np.argmax(predic)))])
	return answer



def main():
	train_data_file = os.path.join(os.path.join(BASE_DIR , "dataset") , "TRAIN_FILE.txt")
	train_data = data.Data(train_data_file , "train")	

	'''
	# trainning data: sentence -> word_id sequence
	tokenizer = Tokenizer(filters = '')
	tokenizer.fit_on_texts(train_data.concate_data)
	sequences = tokenizer.texts_to_sequences(train_data.concate_data)
	word_index = tokenizer.word_index
	with open('tokenizer.pickle' , 'wb') as handle:
		pickle.dump(tokenizer , handle)
	#show_info(sequences) # show the info of the padding
	#pad to the same length
	train_vec = pad_sequences(sequences , maxlen = MAX_SEQUENCE_LENGTH)
	#embedding layer
	embedding_layer = get_embeddinglayer(word_index)

	#training label : one hot encoding
	tokenizer = Tokenizer(filters = '')
	tokenizer.fit_on_texts(train_data.raw_relation)
	sequences = tokenizer.texts_to_sequences(train_data.raw_relation)
	train_label = to_categorical(sequences)
	'''
	#rnn model
	rnn_model = create_rnn_model(train_data.embedding_layer)

	#train
	rnn_model.fit(train_data.sentence_id, train_data.train_label , validation_split = .1 , epochs = 5 , batch_size = 100 )

	#predict
	test_data_file = os.path.join(os.path.join(BASE_DIR , "dataset") , "TEST_FILE.txt")
	test_data = data.Data(test_data_file , "test")
	predict = rnn_model.predict(test_data.sentence_id)
	test_data.one_hot_ans(predict)
	test_data.write("result/rnn_ans.txt")

if __name__ == "__main__":
	main()