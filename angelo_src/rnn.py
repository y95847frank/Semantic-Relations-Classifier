import numpy as np
import data
import os
import pickle
import keras
from keras.layers import Embedding , Dense ,Input , GlobalMaxPooling1D , Bidirectional , LSTM, GRU, Concatenate, Flatten ,Dropout
from keras.models import Model
from keras.models import load_model 
import glob
#import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the current directory
LSTM_UNIT = 256
HYPER_DIM = 64
ROOT_DIM = 64
ENTITY_DIM = 64

def create_rnn_model(embedding_layer):
	sequence_input = Input(shape=(data.MAX_SEQUENCE_LENGTH,), dtype = 'int32')
	embedded_sequences = embedding_layer(sequence_input)

	x = Bidirectional((LSTM(LSTM_UNIT , 
						return_sequences = True ,
						bias_initializer = 'Ones',
						dropout = 0.2,
						recurrent_dropout = 0.2		
													)) , merge_mode = 'sum')(embedded_sequences)

	x = GlobalMaxPooling1D()(x)

	#hypernym
	hypernym_id1 = Input(shape = (1,) , dtype = 'int32')
	hypernym_id2 = Input(shape = (1,) , dtype = 'int32')
	embedding_for_hyper = Embedding(2292, HYPER_DIM)
	embedded_hypernym_id1 = embedding_for_hyper(hypernym_id1)
	embedded_hypernym_id1 = Flatten()(embedded_hypernym_id1)
	embedded_hypernym_id2 = embedding_for_hyper(hypernym_id2)
	embedded_hypernym_id2 = Flatten()(embedded_hypernym_id2)

	#root
	root = Input(shape = (1,) , dtype = 'int32')
	embedding_for_root = Embedding(1263 , ROOT_DIM)
	embedded_root = embedding_for_root(root)
	embedded_root = Flatten()(embedded_root)

	#entity1 , entity2 
	entity1 = Input(shape = (1,) , dtype = 'int32')
	entity2 = Input(shape = (1,) , dtype = 'int32')
	embedding_for_entity = Embedding(5033 , ENTITY_DIM)
	embedded_entity1 = embedding_for_entity(entity1)
	embedded_entity1 = Flatten()(embedded_entity1)
	embedded_entity2 = embedding_for_entity(entity2)
	embedded_entity2 = Flatten()(embedded_entity2)

	preds = keras.layers.Concatenate()([x , embedded_hypernym_id1 , embedded_hypernym_id2 , embedded_root , embedded_entity1 , embedded_entity2])
	preds = Dense( 19 , activation='softmax')(preds)

	model = Model([sequence_input ,hypernym_id1 ,hypernym_id2 , root ,entity1 , entity2] , preds)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['acc'])
	return model 


def main():
	train_data_file = os.path.join(os.path.join(BASE_DIR , "dataset") , "TRAIN_FILE.txt")
	train_data = data.Data(train_data_file , "train")
	quit()

	#rnn model
	rnn_model = create_rnn_model(train_data.embedding_layer)
	print(rnn_model.summary())


	#train
	model_file = 'model/lstm%ssum_drop2_hyper%sd_root%sd_entity%sd'%(LSTM_UNIT , HYPER_DIM , ROOT_DIM , ENTITY_DIM)
	os.makedirs(model_file)
	esCallBack = keras.callbacks.EarlyStopping(monitor='val_acc' , min_delta= 0 , patience = 6 , verbose = 0 , mode = 'auto')
	chCallBack = keras.callbacks.ModelCheckpoint('./%s/weights.{epoch:02d}-{val_acc:.4f}.h5' %(model_file), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	rnn_model.fit([train_data.sentence_id,train_data.hyper_id[:,0],train_data.hyper_id[:,1] , train_data.root_id , train_data.entity1_id , train_data.entity2_id], train_data.train_label , 
					validation_split = .1 , 
					shuffle = True,
					epochs = 30 , 
					batch_size = 100 , 
					callbacks = [esCallBack , chCallBack])

	#predict
	model_file = 'model/glove300d_lstm%ssum_drop2_hyper%sd_root%sd_entity%sd'%(LSTM_UNIT , HYPER_DIM , ROOT_DIM , ENTITY_DIM)
	test_data_file = os.path.join(os.path.join(BASE_DIR , "dataset") , "TEST_FILE.txt")
	test_data = data.Data(test_data_file , "test")

	list_of_files = glob.glob(os.path.join(model_file , "*"))
	latest_file = max(list_of_files , key = os.path.getctime)
	rnn_model = load_model(latest_file)

	predict = rnn_model.predict([test_data.sentence_id,test_data.hyper_id[:,0],test_data.hyper_id[:,1] , test_data.root_id , test_data.entity1_id , test_data.entity2_id])
	test_data.one_hot_ans(predict)
	test_data.write("result/rnn_ans.txt")

if __name__ == "__main__":
	main()