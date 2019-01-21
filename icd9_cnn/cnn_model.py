import tensorflow as tf

# core logic based on http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper

class NNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)        

        
    @with_self_graph
    def SetParams(self, vocab_size, sequence_length, embedding_size, num_classes, learning_rate, filter_sizes,num_filters,l2_reg_lambda=0.0):
        self.vocab_size = vocab_size 
        self.embedding_size =embedding_size 
        self.num_classes =num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        # sequence_length: The length of our sentences. In this example all our sentences 
        #have the same length (59)
        self.sequence_length = sequence_length
        
        self.learning_rate = learning_rate
        
        
        # Training hyperparameters; these can be changed with feed_dict,
        with tf.name_scope("Training_Parameters"):
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)
    
    
        
    @with_self_graph
    def BuildCoreGraph(self):
        
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        #self.x = tf.placeholder(tf.float32,shape=[None,self.sequence_length,self.embedding_size],name="input_x") embedded already 
        self.input_y = tf.placeholder(tf.float32, shape=[None,self.num_classes],  name="input_y")
        
        # Embedding
        # -----------------------------------------------------------------------------
                # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            
        # x embedded SHAPE:   [batch_size, sequence_length, embedding_size]
        
        # TensorFlow convolutional conv2d operation expects a 4-dimensional tensor
        # with dimensions corresponding to batch, width, height and channel. 
        # The result of our embedding does not contain the channel dimension, so we add it manually, 
        self.x_expanded = tf.expand_dims(self.embedded_chars, -1)
        #self.x_expanded .SHAPE: [batch_size, sequence_length, embedding_size, 1]
        
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = [] 
        for i, filter_size in enumerate(self.filter_sizes): 
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # ----------------------------------------------------------------------
                # filter shape: [window_region_height, window_region_width, 
                #                number of input channels, number of filters for each region)
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                
                # Here, W is our filter matrix. Each filter slides over the whole embedding matrix, 
                # but varies in how many words it covers.
                # "VALID" padding means that we slide the filter over our sentence without padding the edges, 
                # performing a narrow convolution that gives us an output of shape 
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")                 
                conv = tf.nn.bias_add(tf.nn.conv2d( self.x_expanded, W, 
                     strides=[1, 1, 1, 1], padding="VALID", name="conv"), b)
                                
                # Apply nonlinearity  
                h = tf.nn.relu(conv, name="relu")   
                # h.SHAPE: [1, sequence_length - filter_size + 1, 1, 1]
                
                # Maxpooling over the outputs
                # ------------------------------------------------------------------ 
                conv_vector_length =  self.sequence_length - filter_size + 1  
                # The pooling ops sweep a rectangular window over the input tensor, computing a reduction operation for each window
                # in this case max. Each pooling op uses rectangular windows of size ksize separated by offset strides   
                k_size = [1, conv_vector_length, 1, 1] # shape of output vector from conv
                pooled = tf.nn.max_pool( h, ksize=k_size,
                     strides=[1, 1, 1, 1], padding='VALID', name="pool") 
                # pooled. SHAPE: [batch_size, 1, 1, num_filters]
                # This is essentially a feature vector, where the last dimension corresponds to our features.
                
                pooled_outputs.append(pooled)
                
        # Combine all the pooled features
        # -----------------------------------------------------------------
        # Once we have all the pooled output tensors from each filter size we combine them into one long feature vector
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # self.h_pool_flat SHAPE: batch_size, num_filters_total]
       
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #self.predictions = tf.argmax(self.scores, 1, name="predictions")
            
            #self.y_hat = tf.sigmoid(self.scores) 
            self.y_hat = tf.nn.softmax(self.scores) 
        
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            
    @with_self_graph
    def BuildTrainGraph(self):        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #optimizer = tf.train.AdadeltaOptimizer (self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)