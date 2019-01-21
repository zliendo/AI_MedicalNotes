from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras import regularizers
import attention_util
from keras import optimizers

''' code based on:
https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/sentiment_cnn.py
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
'''

def build_icd9_cnn_model(input_seq_length, 
                         max_vocab, external_embeddings, embedding_dim, embedding_matrix, 
                         num_filters, filter_sizes, 
                         training_dropout,
                         num_classes,trainable_Embeddings=True):
    #Embedding
    model_input = Input(shape=(input_seq_length, ))
    if external_embeddings:
        # use embedding_matrix 
        z = Embedding(max_vocab + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=input_seq_length,
                            trainable=trainable_Embeddings,embeddings_regularizer=regularizers.l2(0.0001))(model_input)
    else:
        # train embeddings 
        z =  Embedding(max_vocab + 1, 
                   embedding_dim, 
                   input_length=input_seq_length, embeddings_regularizer=regularizers.l2(0.0001),
                   name="embedding")(model_input)

    #z = Dropout(0.1)(z)
    # Convolutional block
    conv_blocks = []
    for i,sz in enumerate(filter_sizes):
        conv = Convolution1D(filters=num_filters,                         
                         kernel_size=sz,
                         padding="valid",kernel_regularizer=regularizers.l2(0.001),
                         activation="relu",
                         strides=1)(z)
        window_pool_size =  input_seq_length  - sz + 1 
        #conv = MaxPooling1D(pool_size=window_pool_size)(conv)  
        words_attention_vector = attention_util.attention_layer(conv, window_pool_size,50,str(i))
        #conv = Flatten()(words_attention_vector)
        conv_blocks.append(words_attention_vector)

    #concatenate
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    #score prediction
    #z = Dense(200, activation="relu")(z)  #to avoid overfitting? I don't think this is necessary
    z = Dropout(training_dropout)(z)
    #model_output = Dense(num_classes, activation="softmax")(z)
    model_output = Dense(num_classes, kernel_regularizer=regularizers.l2(0.0001),
                         activation="sigmoid")(z)

    #creating model
    model = Model(model_input, model_output)
    # what to use for tf.nn.softmax_cross_entropy_with_logits?
    adam_op = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=adam_op, metrics=["accuracy"])
    #model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print model.summary()

    return model