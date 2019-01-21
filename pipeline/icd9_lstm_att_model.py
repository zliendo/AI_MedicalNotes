from keras.models import  Model
from keras.layers import Dense, Dropout, Flatten, Input,  Embedding,Bidirectional
from keras.layers.merge import Concatenate
from keras.layers import LSTM
from keras.layers import  MaxPooling1D, Embedding, Merge, Dropout, LSTM, Bidirectional
from keras.layers.merge import Concatenate
from keras.layers.core import *
from keras.layers import merge, dot, add
from keras import backend as K
import attention_util

def build_lstm_att_model(input_seq_length, 
                         max_vocab, external_embeddings, embedding_trainable, embedding_dim, embedding_matrix,                         
                          training_dropout_keep_prob,num_classes):
    #Embedding
    model_input = Input(shape=(input_seq_length, ))
    if external_embeddings:
        # use embedding_matrix 
        z = Embedding(max_vocab + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=input_seq_length,
                            trainable=embedding_trainable,name = "Embeddng")(model_input)
    else:
        # train embeddings 
        z = Embedding(max_vocab + 1, 
                   embedding_dim, 
                   input_length=input_seq_length, 
                   name="Embedding")(model_input)

    # LSTM
    lstm_units= 50
    l_lstm = LSTM(lstm_units,return_sequences=True)(z)
    
    #attention
    words_attention_vector = attention_util.attention_layer(l_lstm,input_seq_length,lstm_units) 
    
    #score prediction 
    z = Dropout(training_dropout_keep_prob)(words_attention_vector)
    model_output = Dense(num_classes, activation="sigmoid", name="Output_Layer")(z)

    #creating model
    model = Model(model_input, model_output)
    # what to use for tf.nn.softmax_cross_entropy_with_logits?
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print model.summary()

    return model
