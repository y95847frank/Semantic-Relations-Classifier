import re
import string
import pickle
import os
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding 
from nltk.corpus import wordnet as wn
import spacy

GLOVE_DIR = "/Users/angelocsc/Desktop/NTU/NLPLab/glove.6B"
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 25

def show_info(sequences):
	avg_len = 0
	max_len = 0
	number = 0
	for sequence in sequences:
		if(len(sequence) > 25):
			print(number+1 , len(sequence))
		avg_len += len(sequence)
		number += 1
		if(len(sequence) > max_len):
			max_len = len(sequence)
	print("number %d" %(number))
	print("avg len: %d" %(avg_len/number))
	print("max len: %d" %(max_len))
	return

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

class Data():
	def __init__(self, data_path , data_type):
		self.data_type = data_type # train or test 
		self.data_id = list() # id of a text
		self.raw_data = list() #!!<e1>This</e1> is an <e2>example</e2>
		self.anonymize_data = list() #!!entity1 is an entity2
		self.preprocess_data = list() #entity1 is an entity2
		self.entity1 = list()
		self.entity2 = list()
		self.hypernym = list() #[entity_hypernym1 , entity_hypernym2]
		self.hyper_id = list()
		self.root = list()
		self.root_id = list()
		self.sentence_id = list() # [0,0,0,......0,1,2,3,4]
		self.concate_data = list() # (only get 8 words before and after entity)
		self.raw_relation = list() #Component-Whole(e2,e1)
		self.relation = list() #Component-Whole
		self.relation_dir = list() #1
		self.train_label = list() # [0,0,0,0,0,0,0,0,0,0,1,0...0] one-hot encoding

		if(self.data_type == "train"):
			with open(data_path , 'r') as f:
				lines = f.read().strip().split('\n')
				for i , line in enumerate(lines):
					if(i % 4 == 0):
						data = line.split('\t')[1][1:-2]
						self.extract_data(data)

					elif(i % 4 == 1):
						self.raw_relation.append(line)
						self.relation.append(line.split('(')[0])
						if(line.split('(')[0] == "Other"):
							self.relation_dir.append(-1) # other has no direction
						elif(line.split('(')[1].split(',')[0] == "e1"):
							self.relation_dir.append(0) # 0 represent (e1,e2)
						else:
							self.relation_dir.append(1) # 1 represent (e2,e1)
			self.sen2id()
			self.hyper2id()
			#load root list
			with open('root_train.pickle'  , 'rb') as fp:
				self.root = pickle.load( fp)
			self.root2id()
			self.entity2id()
			self.one_hot_relation()

		elif(self.data_type == "test"):
			with open(data_path , 'r') as f:
				lines = f.read().strip().split('\n')
				for line in lines:
					[key , data] = line.split('\t')
					self.data_id.append(key)
					self.extract_data(data[1:-2])
			self.sen2id()
			self.hyper2id()
			#load root list
			with open('root_test.pickle'  , 'rb') as fp:
				self.root = pickle.load( fp)
			self.root2id()
			self.entity2id()

	def extract_data(self , data):
		data = re.sub(r'\sa\s|\sthe\s|\san\s' , ' ' ,data)

		entity1 = data.split('<e1>')[1].split('</e1>')[0].split()[-1]
		entity2 = data.split('<e2>')[1].split('</e2>')[0].split()[-1]

		anonymize_data = re.sub(r'<e1>.*</e1>' , 'entity1' , data)
		anonymize_data = re.sub(r'<e2>.*</e2>' , 'entity2' , anonymize_data)
		self.anonymize_data.append(anonymize_data)

		raw_data = re.sub(r'<e1>' , '<e1> ' , data)
		raw_data = re.sub(r'</e1>' , ' <e1>' , raw_data)
		raw_data = re.sub(r'<e2>' , '<e2> ' , raw_data)
		raw_data = re.sub(r'</e2>' , ' <e2> ' , raw_data)
		self.raw_data.append(raw_data)
		self.preprocess_data.append(self.preprocess(data))
		self.concate_data.append(self.concate(data , 8))

		#morphy ex: dogs -> dog
		if(wn.morphy(entity1)):
			entity1 = wn.morphy(entity1)
		if(wn.morphy(entity2)):
			entity2 = wn.morphy(entity2)

		self.entity1.append(entity1)
		self.entity2.append(entity2)

		if(len(wn.synsets(entity1)) == 0 or len(wn.synsets(entity1)[0].hypernyms()) == 0):
			hypernym1 = "unknown"
		else:
			hypernym1 = wn.synsets(entity1)[0].hypernyms()[0].name().split('.')[0]
			#if(wn.synsets(entity1)[0].hypernyms()[0].name().split('.')[1] != 'n'):
			#	print(len(self.raw_data) ,hypernym1  , wn.synsets(entity1)[0].hypernyms()[0].name().split('.')[1])
		if(len(wn.synsets(entity2)) == 0 or len(wn.synsets(entity2)[0].hypernyms()) == 0):
			hypernym2 = "unknown"
		else:
			hypernym2 = wn.synsets(entity2)[0].hypernyms()[0].name().split('.')[0]
			#if(wn.synsets(entity2)[0].hypernyms()[0].name().split('.')[1] != 'n'):
			#	print(len(self.raw_data) , hypernym2,   wn.synsets(entity2)[0].hypernyms()[0].name().split('.')[1])
		self.hypernym.append(' '.join([hypernym1 , hypernym2]))

		#get root
		'''
		nlp = spacy.load('en_core_web_sm')
		doc = nlp(u'%s' %(re.sub(r'<e1>|</e1>|<e2>|</e2>' , '' , data)))
		for token in doc:
			if(token.dep_ == 'ROOT'):
				self.root.append(token.lemma_)
				print(len(self.raw_data) ,count, token.lemma_)
				break;
		'''
		return

	def preprocess(self, sentence): # lower & delete punctuation & delete digits
		splits = re.split(r'<|>' , sentence)
		words_before = splits[0].lower().translate(str.maketrans('','',string.punctuation+string.digits))
		entity1 = '<e1> ' + splits[2] + ' </e1>'
		words_between = splits[4].lower().translate(str.maketrans('','',string.punctuation+string.digits))
		entity2 = '<e2> ' + splits[6] + ' </e2>'
		words_after = splits[8].lower().translate(str.maketrans('','',string.punctuation+string.digits))
		#print(words_before , entity1 , words_between , entity2 , words_after)
		return words_before + entity1 + words_between + entity2 + words_after

	def concate(self, sentence , number):
		splits = re.split(r'<|>' , sentence)
		words_before = splits[0].lower().translate(str.maketrans('','',string.punctuation+string.digits))
		entity1 = ' <e1> ' + splits[2] + ' </e1>'
		words_between = splits[4].lower().translate(str.maketrans('','',string.punctuation+string.digits))
		entity2 = '<e2> ' + splits[6] + ' </e2> '
		words_after = splits[8].lower().translate(str.maketrans('','',string.punctuation+string.digits))
		words_before = ' '.join(words_before.split()[-number:])
		words_after = ' '.join(words_after.split()[-number:])
		return words_before + entity1 + words_between + entity2 + words_after

	def sen2id(self):
		if(self.data_type == "train"):
			tokenizer = Tokenizer(filters = '')
			tokenizer.fit_on_texts(self.concate_data)
			with open('tokenizer.pickle' , 'wb') as handle:
				pickle.dump(tokenizer , handle)
		elif(self.data_type == "test"):
			with open('tokenizer.pickle' , 'rb') as handle:
				tokenizer = pickle.load(handle)
		sequences = tokenizer.texts_to_sequences(self.concate_data)
		word_index = tokenizer.word_index
		#show_info(sequences) # show the info of the padding
		#pad to the same length
		self.sentence_id = pad_sequences(sequences , maxlen = MAX_SEQUENCE_LENGTH)
		#embedding layer
		self.embedding_layer = get_embeddinglayer(word_index)

	def hyper2id(self):
		if(self.data_type == "train"):
			tokenizer = Tokenizer(filters = '')
			tokenizer.fit_on_texts(self.hypernym)
			with open('tokenizer_hyper.pickle' , 'wb') as handle:
				pickle.dump(tokenizer , handle)
		elif(self.data_type == "test"):
			with open('tokenizer_hyper.pickle' , 'rb') as handle:
				tokenizer = pickle.load(handle)
		sequences = tokenizer.texts_to_sequences(self.hypernym)
		sequences = pad_sequences(sequences , maxlen = 2)
		self.hyper_id = np.array(sequences)

	def root2id(self):
		if(self.data_type == "train"):
			tokenizer = Tokenizer(filters = '')
			tokenizer.fit_on_texts(self.root)
			with open('tokenizer_root.pickle' , 'wb') as handle:
				pickle.dump(tokenizer , handle)
		elif(self.data_type == "test"):
			with open('tokenizer_root.pickle' , 'rb') as handle:
				tokenizer = pickle.load(handle)
		sequences = tokenizer.texts_to_sequences(self.root)
		sequences = pad_sequences(sequences , maxlen = 1)
		self.root_id = np.array(sequences)

	def entity2id(self):
		if(self.data_type == "train"):
			tokenizer = Tokenizer(filters = '')
			tokenizer.fit_on_texts(self.entity1+self.entity2)
			with open('tokenizer_entity.pickle', 'wb') as handle:
				pickle.dump(tokenizer , handle)
		elif(self.data_type == "test"):
			with open('tokenizer_entity.pickle' , 'rb') as handle:
				tokenizer = pickle.load(handle)
		entity1_sequences = tokenizer.texts_to_sequences(self.entity1)
		entity2_sequences = tokenizer.texts_to_sequences(self.entity2)
		entity1_sequences = pad_sequences(entity1_sequences , maxlen = 1)
		entity2_sequences = pad_sequences(entity2_sequences , maxlen = 1)
		self.entity1_id = np.array(entity1_sequences)
		self.entity2_id = np.array(entity2_sequences)

	def one_hot_relation(self):
		tokenizer = Tokenizer(filters = '' , lower = False)
		tokenizer.fit_on_texts(self.raw_relation)
		sequences = tokenizer.texts_to_sequences(self.raw_relation)
		print(tokenizer.word_index)
		self.train_label = to_categorical(sequences)
		with open('tokenizer_label.pickle' , 'wb') as handle:
				pickle.dump(tokenizer , handle)

	def one_hot_ans(self,prediction):
		self.answer = list()
		with open('tokenizer_label.pickle' , 'rb') as handle:
				tokenizer = pickle.load( handle)
		word_index = tokenizer.word_index
		for predic in prediction:
			self.answer.append(list(word_index.keys())[list(word_index.values()).index(np.argmax(predic))])
		
	def write(self,out_path):
		with open(out_path , 'w') as f:
			for i , ans in enumerate(self.answer):
				f.write(self.data_id[i])
				f.write('\t')
				f.write(ans)
				f.write('\n')

