import tensorflow as tf


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
    def SetParams(self,  Hidden_dims, learning_rate, vocabulary_size, y_dim):
        # Model structure; these need to be fixed for a given model.
        self.Hidden_dims = Hidden_dims
        self.learning_rate = learning_rate
        self.V =vocabulary_size  
        self.y_dim = y_dim
        
    @with_self_graph    
    def affine_layer(self, hidden_dim, x, seed=0):
        self.W = tf.get_variable("W", initializer=tf.contrib.layers.xavier_initializer(seed = seed),  \
                        trainable=True,shape=[x.shape[1],hidden_dim])        
        self.b = tf.get_variable("b", initializer=tf.zeros_initializer(), \
                        trainable=True,shape=[hidden_dim])
        return tf.matmul(x,self.W) + self.b 

    @with_self_graph
    def fully_connected_layers(self,x):
        for i in range(len(self.Hidden_dims)):
            with tf.variable_scope("layer_" + str(i)):
                x = tf.nn.relu(self.affine_layer(self.Hidden_dims[i], x))
        return x
    @with_self_graph
    def BuildCoreGraph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.V])
        self.target_y = tf.placeholder(tf.float32, shape=[None,None])
        
        z = self.fully_connected_layers(self.x)        
        
        self.y_logit = tf.squeeze(self.affine_layer(self.y_dim,z))
        self.y_hat = tf.sigmoid(self.y_logit)  
     
        self.loss = tf.reduce_mean (tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_y, logits=self.y_logit))
        
    @with_self_graph
    def BuildTrainGraph(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.loss)      
