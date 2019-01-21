from keras.models import  Model
from keras.layers import Dense, Dropout, Flatten, Input,  Embedding,Bidirectional
from keras.layers.merge import Concatenate
from keras.layers import LSTM


def build_lstm_model(input_seq_length, 
                         max_vocab, external_embeddings, embedding_trainable, embedding_dim, embedding_matrix,
                     training_dropout_keep_prob, num_classes):
    #Embedding
    model_input = Input(shape=(input_seq_length, ))
    if external_embeddings:
        # use embedding_matrix 
        z = Embedding(max_vocab + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=input_seq_length,
                            trainable=embedding_trainable)(model_input)
    else:
        # train embeddings 
        z =  Embedding(max_vocab + 1, 
                   embedding_dim, 
                   input_length=input_seq_length, 
                   name="embedding")(model_input)

    # LSTM
    l_lstm = LSTM(50)(z)
    
    z = Dropout(training_dropout_keep_prob)(l_lstm)
    
    #score prediction 
    model_output = Dense(num_classes, activation="sigmoid")(z)

    #creating model
    model = Model(model_input, model_output)
    # what to use for tf.nn.softmax_cross_entropy_with_logits?
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print model.summary()

    return model