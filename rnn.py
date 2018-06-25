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
MAX_SEQUENCE_LENGTH = 25

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


def main():
	train_data_file = os.path.join(os.path.join(BASE_DIR , "dataset") , "TRAIN_FILE.txt")
	train_data = data.Data(train_data_file , "train")	

	#rnn model
	rnn_model = create_rnn_model(train_data.embedding_layer)

	#train
	rnn_model.fit(train_data.sentence_id, train_data.train_label , validation_split = .1 , epochs = 7 , batch_size = 100 )

	#predict
	test_data_file = os.path.join(os.path.join(BASE_DIR , "dataset") , "TEST_FILE.txt")
	test_data = data.Data(test_data_file , "test")
	predict = rnn_model.predict(test_data.sentence_id)
	test_data.one_hot_ans(predict)
	test_data.write("result/rnn_ans.txt")

if __name__ == "__main__":
	main()