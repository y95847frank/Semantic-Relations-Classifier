import keras

from keras.layers import Input, Conv1D, Dropout, Merge, Dense, Activation, Dense, Flatten, Reshape
from keras.models import Model
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate

class CNN:
    def __init__(self, option):
        self.options = option

    def build_model(self):
        # maxset_len = 100, max_pos = 52 (30+22), pos_dim = 50
        
        #input1 is for position1 embedding layer
        input1 = Input(shape = (self.options['maxsen_len'], ))
        x = Embedding(self.options['max_pos'], self.options['pos_dim'], input_length = self.options['maxsen_len'])(input1)
        
        #input2 is for position2 embedding layer
        input2 = Input(shape = (self.options['maxsen_len'], ))
        y = Embedding(self.options['max_pos'], self.options['pos_dim'], input_length = self.options['maxsen_len'])(input2)

        #input3 is for word embedding layer
        input3 = Input(shape = (self.options['maxsen_len'], ))
        z = Embedding(self.options['vocab_size'], self.options['emb_dim'], input_length = self.options['maxsen_len'], weights = [self.options['embedding']], trainable=False)(input3)
        

        #input4 = Input(shape = (1, ))
        #a = Embedding(self.options['POS_size'], self.options['emb_dim'], input_length = 1)(input4)

        #input5 = Input(shape = (1, ))
        #b = Embedding(self.options['POS_size'], self.options['emb_dim'], input_length = 1)(input5)
        
        input6 = Input(shape = (8, ))
        c = Embedding(self.options['vocab_size'], self.options['emb_dim'], input_length = 8, weights = [self.options['embedding']], trainable=False)(input6)

        merge = concatenate([x, y, z])

        submodel = Conv1D(self.options['num_filter'], 2, padding='valid', input_shape=(self.options['maxsen_len'], self.options['emb_dim'] + self.options['pos_dim'] * 2 ))(merge)
        
        submodel = GlobalMaxPooling1D()(submodel)

        submodel = Reshape((1, 500))(submodel)

        submodel = Dense(300, activation='tanh')(submodel)
        
        final_merge = concatenate([c, submodel], axis = 1)

        final_merge = Flatten()(final_merge)
        
        #final_merge = Dropout(0.25)(final_merge)
        
        #output = Dense(128, activation='relu')(final_merge)
        output = Dense(self.options['n_class'], activation='softmax')(final_merge)


        return Model(inputs=[input1, input2, input3, input6], outputs = output)
        
