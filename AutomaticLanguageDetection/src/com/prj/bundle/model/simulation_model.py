'''
Created on Jun 21, 2021

@author: Neha Warikoo
'''

import sys, os, re
import tensorflow as tf
from tensorflow.keras import models, layers
import collections

class SequentialMultiClassClassifier(object):
    
    def __init__(self, unigram_tensor, ascii_tensor, ascii_group_feature, label, hidden_dim, 
                 unigram_featuremap_size, symbol_group_size, is_training):
        
        
        self.hidden_dim = hidden_dim
        self.unigram_featuremap_size = unigram_featuremap_size
        #self.bigram_featuremap_size = bigram_featuremap_size
        self.unigram_embedding = None
        self.bigram_embedding = None
        
        with tf.compat.v1.variable_scope('smc'):
            with tf.compat.v1.variable_scope('embedding'):
                self.unigram_output, self.unigram_embedding = embedding_lookup(
                    embedding_name='unigram_embedding',
                    input_tensor = unigram_tensor, 
                    embedding_size = self.hidden_dim, 
                    featuremap_size = self.unigram_featuremap_size)
                
                self.ascii_output = tf.expand_dims(
                    tf.cast(ascii_tensor, dtype=tf.float32), axis=-1)
                self.ascii_group_output = tf.one_hot(
                    ascii_group_feature, depth=symbol_group_size)
                '''
                self.ascii_group_output = tf.expand_dims(
                    tf.cast(ascii_group_feature, dtype=tf.float32), axis=-1)
                '''
                tf.compat.v1.logging.info(self.ascii_output)
                tf.compat.v1.logging.info(self.ascii_group_output)
                
                self.hidden_output = tf.concat(
                    [self.unigram_output, self.ascii_group_output],axis=-1)
                hidden_shape = get_tensor_shape(self.hidden_output)
                tf.compat.v1.logging.info(self.hidden_output)
                
            features_vector = []
            filter_size = 24
                  
            #===================================================================
            # define Convolutions over 2,3,4 filter size to cover small 2,3,4 grams features
            # feature represented as such help study local context associations        
            #===================================================================
            kernel_size = [2,3,4]
            stride_size = 1
            with tf.compat.v1.variable_scope("cnn_unit"):
                for layer_id in kernel_size:
                    with tf.compat.v1.variable_scope("layer_%d" %layer_id):
                        self.cnn_layer_output = tf.keras.layers.Conv1D(
                            filters=64*(filter_size), kernel_size=layer_id, strides=stride_size, 
                            padding='valid', use_bias=True, activation='relu')(
                                self.hidden_output)
                        
                        self.cnn_layer_output = tf.keras.layers.MaxPool1D(
                            pool_size=layer_id, strides=stride_size , padding='valid')(self.cnn_layer_output)
                        self.cnn_layer_output = tf.keras.layers.BatchNormalization()(
                            
                            self.cnn_layer_output)
                        features_vector.append(self.cnn_layer_output)
            
            #===================================================================
            # multi-layer dense network            
            #===================================================================
            dnn_feature_vector = []
            with tf.compat.v1.variable_scope("dense_unit"):
                for layer_feature in features_vector:
                    self.dnn_layer_output = layer_feature
                    for red_index in range(0,4):
                        self.dnn_layer_output = tf.keras.layers.Dense(
                            units=int(64*(filter_size-(6*red_index))), activation='relu')(self.dnn_layer_output)
                    
                    self.dnn_layer_output = tf.keras.layers.Dense(
                            units=int(64), activation='relu')(self.dnn_layer_output)
                    dnn_feature_vector.append(self.dnn_layer_output)
            
            #===================================================================
            # output pooling from all feature vectors        
            #===================================================================
            with tf.compat.v1.variable_scope("output_pool"):
                self.pool_output = tf.concat(
                    dnn_feature_vector, axis=1)
                self.pool_output = tf.keras.layers.Dense(
                    units=int(8), activation='relu')(self.pool_output)
                self.pool_output = tf.keras.layers.Flatten()(self.pool_output)
                
    def get_pool_output(self):
        return(self.pool_output)

    def get_hidden_output(self):
        return(self.hidden_output)

def rnn_cell(hidden_dim):
    return tf.keras.layers.LSTMCell(units=hidden_dim)
        
def get_tensor_shape(tensor):
    
    tensor_shape = tensor.shape.as_list()
    
    dynamic_shape=[]
    for (index, dim) in enumerate(tensor_shape):
        if dim is None:
            dynamic_shape.append(index)
    
    if not dynamic_shape:
        return(tensor_shape)
    
    dynamic_tensor_shape = tf.shape(tensor)
    for index in dynamic_shape:
        tensor_shape[index] = dynamic_tensor_shape[index]
        
    return(tensor_shape)

def create_initializer(initializer_range=0.02):
    return (tf.compat.v1.truncated_normal_initializer(stddev=initializer_range))
        
def embedding_lookup(embedding_name, input_tensor, embedding_size, 
                         featuremap_size, initializer_range=0.02):
    
    embedding_table = tf.compat.v1.get_variable(
        name=embedding_name, 
        shape=[featuremap_size, embedding_size], 
        initializer=create_initializer(initializer_range))
    
    output = tf.nn.embedding_lookup(embedding_table, input_tensor)
    input_shape = get_tensor_shape(input_tensor)
    output = tf.reshape(output, [input_shape[0],input_shape[-1] , embedding_size])
    
    return(output, embedding_table)

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    
    """Compute the union of the current variables and checkpoint variables."""
    
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        
        if name == 'output_weights:0' or name == 'output_bias:0':
            continue
        
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
            
    return(assignment_map, initialized_variable_names)





