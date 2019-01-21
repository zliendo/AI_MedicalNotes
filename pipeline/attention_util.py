from keras.layers import Dense,  Input
from keras.layers import  Merge,TimeDistributed
from keras.layers.merge import Concatenate
from keras.layers.core import *
from keras.layers import merge, dot, add
from keras import backend as K
# based on paper: Hierarchical Attention networks for document classification
# starting code from:
# https://groups.google.com/forum/#!msg/keras-users/IWK9opMFavQ/AITppppfAgAJ

# note: there is a lot of sample codes in the internet that do not work, and their authors do mention that, 
# they don't see a difference when applying the attention mechanism
#
# I did have to review closely the formulas presented on the papers about Attention to figure it out what type of
# code will actually work

def attention_layer(inputs, TIME_STEPS,lstm_units, i='1'):

    # inputs.shape = (batch_size, time_steps, input_dim)
    #(3) u_it: we first feed the word annotation through a one-layer MLP to get the hidden representation u_it
    inputs= Dropout(0.5)(inputs)
    u_it = TimeDistributed(Dense(lstm_units, activation='tanh',
                                 kernel_regularizer=regularizers.l2(0.0001),
                                 name='u_it'+i))(inputs)

    u_it= Dropout(0.5)(u_it)
    # (4) alpha_it: then we measure the importance of x as the similarity of u_it with a x level
    # context vector u_w and get a normalized importance weight alpha_it through a softmax function
    # The word context vector uw is randomly initialized and jointly learned during the training process.
    #alpha_it  = TimeDistributed(Dense(TIME_STEPS, activation='softmax',use_bias=False))(u_it)
    att = TimeDistributed(Dense(1, 
                                kernel_regularizer=regularizers.l2(0.0001),
                                bias=False))(u_it)                         
    att = Reshape((TIME_STEPS,))(att)                                                       
    att = Activation('softmax', name='alpha_it_softmax'+i)(att) 

    
    # (5) s_i: After that, we compute the sentence vector s_i 
    #     as a weighted sum of the word annotations based on the weights alpha_it.
    s_i =merge([att, inputs], mode='dot', dot_axes=(1,1), name='s_i_dot'+i) 
    
    
    return s_i