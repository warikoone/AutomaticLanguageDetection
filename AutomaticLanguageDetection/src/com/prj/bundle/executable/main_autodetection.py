'''
Created on Jun 20, 2021

@author: Neha Warikoo
'''

import sys, os, re, six
import tensorflow as tf
import nltk
import collections
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from tensorflow.keras import layers 
from distributed.config import config

sys_path = re.sub('\/src\/com\/prj\/bundle\/executable','',os.getcwd())
sys.path.insert(1,sys_path)

from src.com.prj.bundle.model import simulation_model

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("do_train", False, "run for training setup")

flags.DEFINE_bool("do_eval", False, "run on development setup")

flags.DEFINE_bool("do_predict", False, "predict test data")

flags.DEFINE_string('type', 'sentence', 'define type of language detection : sentence/document')

flags.DEFINE_string('eval_type', 'validation', 
                    '2 options: a)validation - for n-fold model training/eval/test purpose and b)predict - for prediction')

flags.DEFINE_string('resource',None,'resource path')

flags.DEFINE_string('output',None,'model output path')

flags.DEFINE_integer('seq_length', 800, 'maximum length of sequence')

flags.DEFINE_integer('hidden_dim', 768, 'hidden dimension size')

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 100,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint from training model.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

class InputFrame(object):
    
    def __init__(self, doc_id, sentence, label):
        self.doc_id = doc_id
        self.sentence = sentence
        self.label = label
        
#===============================================================================
# process raw input data : format as InputProcessor params 
#===============================================================================
class InputProcessor(object):
    
    def __init__(self):
        print('init input processor')
    
    def get_train_data(self, resource_dir, seq_length):
        buffer_df = pd.read_csv(os.path.join(resource_dir,'train.tsv'), sep='\t')
        inputs = self.generate_examples(buffer_df, False, seq_length)
        return(inputs)
    
    def get_eval_data(self, resource_dir, seq_length):
        buffer_df = pd.read_csv(os.path.join(resource_dir,'eval.tsv'), sep='\t')
        inputs = self.generate_examples(buffer_df, False, seq_length)
        return(inputs)
    
    def get_predict_data(self, resource_dir, seq_length):
        buffer_df = pd.read_csv(os.path.join(resource_dir,'test.tsv'), sep='\t')
        inputs = self.generate_examples(buffer_df, True, seq_length)
        return(inputs)
    
    def generate_examples(self, buffer_df, pred_stat, seq_length):
        inputs = []
        for i in range(buffer_df.shape[0]): #500
            doc_id = buffer_df.loc[i,'doc_id']
            sentence = buffer_df.loc[i,'sentence']
            label = 'english'
            if len(str(sentence)) > seq_length:
                continue  
            if not pred_stat:
                label = buffer_df.loc[i,'label']
            inputs.append(InputFrame(doc_id, sentence, label))
        return(inputs)
    
#===============================================================================
# Define input features to be used with moel
#===============================================================================
class InputFeatures(object):
    
    def __init__(self, unigram_feature, ascii_feature, ascii_group_feature, label):
        self.unigram_feature = unigram_feature
        self.ascii_feature = ascii_feature
        self.ascii_group_feature = ascii_group_feature
        self.label = label

#===============================================================================
# linguistic features defined with ascii grouping        
#===============================================================================
symbol_group = {
        0:(0, 190), 1:(190, 382), 2:(383, 687), 3:(688,903), 4:(904, 1375), 5:(1376, 1566),
        6:(1567,2303), 7:(2304, 2431), 8:(2432, 2564), 9:(2565, 2820), 10:(2821, 6100),
        11:(6101, 7544), 12:(7545, 8208), 13:(8209, 10240), 14:(10241, 10495), 
        15:(10496, 12532), 16:(12353, 12991), 17:(12992, 40959), 18:(40960, 42191), 
        19:(42192, 65535), 20:(65536, 0)}
        
def _extract_inline_features_(text, regex_punc_pattern):
        
        text = str(text).lower().strip()
        text = re.sub(regex_punc_pattern, '', text)
        text = re.sub('\d+','', text).strip()
        char_unigrams = list(map(lambda val: str(val),list(text)))
        return(char_unigrams)
    
def generate_punct_bigrams(punct_chars):
    punct_bigrams = []
    for i in punct_chars:
        punct_bigrams.extend(list(map(lambda val:str(i)+'-'+str(val), punct_chars)))
    return(punct_bigrams)
    
def _map_representation_(char_feature, mapper, seq_length, type):
    
    def get_index(each_feature):
        map_value = None
        condition = mapper.loc[:,'char'] == each_feature
        map_index = mapper.index[condition].to_list()
        if len(map_index) == 1:
            map_value = mapper.loc[map_index[0],'id']
        else:
            tf.compat.v1.logging.info('recheck occurrence of %s' %each_feature)
            sys.exit()
            
        return(map_value)
    
    feature_map = []
    #print(char_feature)
    for each_feature in char_feature:
        #print(each_feature)
        map_value = get_index(each_feature)
        feature_map.append(map_value)
            
    if len(feature_map) > seq_length:
        return None
    else:
        difference = seq_length - len(feature_map)
        if difference > 0:
            map_value = get_index('#')
            #===================================================================
            # if type == 'uni':
            #     map_value = get_index('#')
            # elif type == 'bi':
            #     map_value = get_index(generate_punct_bigrams(['#'])[0])
            #===================================================================
            append_tag = [map_value]*difference
            feature_map.extend(append_tag)
            
    #feature_map = list(map(lambda val:np.float32(val), feature_map))
    return(feature_map)

def categorize_as_symbolgroups(feature_list):

    def chk_group_range(val):
        for key, value in six.iteritems(symbol_group):
            if value[1] == 0:
                if val >= value[0]:
                    return key
            else:
                if (val >= value[0] and val <= value[1]):
                    return key
    
    feature_value = list(map(lambda val : chk_group_range(val), feature_list)) 
    return(feature_value)
    
def map_to_features(input_value, seq_length, resource_dir, label_map, regex_punc_pattern, 
                    unigram_mapper):
    #===========================================================================
    # map input sentences to InputFeature type 
    #===========================================================================
    char_unigrams = _extract_inline_features_(
        input_value.sentence, regex_punc_pattern)
    unigram_feature = _map_representation_(
        char_unigrams, unigram_mapper, seq_length,'uni')
    ascii_feature = list(map(lambda val : np.int64(ord(val)), char_unigrams))
    if len(char_unigrams) < seq_length:
        diff = seq_length - len(char_unigrams)
        ascii_feature.extend([ord('#')]*diff)
    ascii_group_feature = categorize_as_symbolgroups(ascii_feature) 
    ascii_group_feature = list(map(lambda val : np.int64(val), ascii_group_feature))
    #===========================================================================
    # bigram_feature = _map_representation_(
    #     char_bigrams, bigram_mapper, seq_length, 'bi')
    #===========================================================================
    label = label_map[input_value.label]
    #===========================================================================
    # print(unigram_feature)
    # print(ascii_feature)
    # print(ascii_group_feature)
    # print(label)
    #===========================================================================
    if (unigram_feature is None):
        feature = InputFeatures(None, None, None, None)
    else:
        feature = InputFeatures(unigram_feature, ascii_feature, ascii_group_feature, label)
    
    return(feature)
    
def map_input_to_features(inputs, seq_length, record_file, resource_dir, label_map, 
                          regex_punc_pattern, unigram_mapper):
    
    #===========================================================================
    # convert input sentences to mapped features
    #===========================================================================
    fileWriter = tf.io.TFRecordWriter(record_file)
    for i_index, i_value in enumerate(inputs):
        if i_index % 1000 == 0:
            tf.compat.v1.logging.info('writing %d input out of %d' %(i_index, len(inputs)))
            
        feature = map_to_features(i_value, seq_length, resource_dir, label_map, 
                                  regex_punc_pattern, unigram_mapper)
        
        def create_tensor_feature(feature_values):
            
            if isinstance(feature_values, list):
                if any(list(map(lambda currval: isinstance(currval, np.int64), list(feature_values)))):
                    f = tf.train.Feature(int64_list = tf.train.Int64List(value=list(feature_values)))
                elif any(list(map(lambda currval: isinstance(currval, np.float32), list(feature_values)))):
                    f = tf.train.Feature(float_list = tf.train.FloatList(value=list(feature_values)))
                elif any(list(map(lambda currval: isinstance(currval, str), list(feature_values)))):
                    f = tf.train.Feature(bytes_list = tf.train.BytesList(value=list(str(feature_values).encode('utf-8'))))
            elif isinstance(feature_values, int):
                f = tf.train.Feature(int64_list = tf.train.Int64List(value=[feature_values]))
            return(f)
            
        if feature.unigram_feature is not None: 
            features = collections.OrderedDict()
            features['unigram_feature'] = create_tensor_feature(feature.unigram_feature)
            features['ascii_feature'] = create_tensor_feature(feature.ascii_feature)
            features['ascii_group_feature'] = create_tensor_feature(feature.ascii_group_feature)
            features['label'] = create_tensor_feature(feature.label)
                
            tf_example = tf.train.Example(features = tf.train.Features(feature=features))
            fileWriter.write(tf_example.SerializeToString())
    
    fileWriter.close()

    return()

def input_feature_builder(input_file, seq_length, is_training, drop_remainder):
    
    features_map = {
        'unigram_feature' : tf.io.FixedLenFeature([seq_length], tf.int64),
        'ascii_feature' : tf.io.FixedLenFeature([seq_length], tf.int64),
        'ascii_group_feature' : tf.io.FixedLenFeature([seq_length], tf.int64),
        'label' : tf.io.FixedLenFeature([],dtype=tf.int64)
        } 
    
    def _decode_record(record, name_to_features):
        
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int64)
            if t.dtype == np.float32:
                t = tf.cast(t, dtype=tf.float32)
            example[name] = t
            
        return example
    
    def input_fn(params):
        tf.compat.v1.logging.info('**loading from the tensor file record**')
        batch_size = params['batch_size']
        
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size= 100)
            
        d = d.apply(tf.compat.v1.data.experimental.map_and_batch
                                  (lambda record: _decode_record(record, features_map),
                                   batch_size=batch_size,
                                   drop_remainder = drop_remainder))
        return(d)
    
    return(input_fn)

def create_initializer(initializer_range=0.02):
    return (tf.compat.v1.truncated_normal_initializer(stddev=initializer_range))

# call for model, compute per example logits and total loss  
def launch_model(unigram_tensor, ascii_tensor, ascii_group_feature, label, 
                 hidden_dim, num_label, unigram_featuremap_size, symbol_group_size, 
                 is_training):
    
    model = simulation_model.SequentialMultiClassClassifier(
        unigram_tensor, ascii_tensor, ascii_group_feature, label, hidden_dim, 
        unigram_featuremap_size, symbol_group_size, is_training)
    
    output_layer = model.get_pool_output()
    true_output_prob = tf.one_hot(label, depth=num_label)
    
    with tf.compat.v1.variable_scope('loss'):
        logits = tf.keras.layers.Dense(units=num_label)(output_layer)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probabilities = tf.nn.log_softmax(probabilities, axis=-1)
        per_example_loss = -tf.reduce_sum(true_output_prob * log_probabilities, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        
        #probabilities = tf.math.argmax(logits, axis=1)
    return(loss,per_example_loss, probabilities, logits)

# model call, optimization operation
def model_build(seq_length, learning_rate, unigram_featuremap_size, symbol_group_size,
                hidden_dim, num_label, num_train_steps, num_warmup_steps, init_checkpoint):
    
    def model_fn(features, labels, mode, params):
        
        unigram_tensor = features['unigram_feature']
        ascii_tensor = features['ascii_feature']
        ascii_group_feature = features['ascii_group_feature']
        label = features['label']
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        total_loss, per_example_loss, probabilities, logits = launch_model(
            unigram_tensor, ascii_tensor, ascii_group_feature, label, hidden_dim, num_label,
            unigram_featuremap_size, symbol_group_size, is_training)
        
        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = simulation_model.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
            
        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)
            
        if mode == tf.estimator.ModeKeys.TRAIN:
            
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate)
            
            train_op = optimizer.minimize(
                total_loss,
                global_step = tf.compat.v1.train.get_global_step())
            
            accuracy = tf.reduce_mean(
                tf.cast(tf.math.equal(label, tf.math.argmax(logits, axis=1)), tf.float32))
            logging_hook = tf.compat.v1.train.LoggingTensorHook(
                {"loss": total_loss,"accuracy:":accuracy}, every_n_iter=1)
            
            output_spec = tf.estimator.EstimatorSpec(
                mode,
                loss=total_loss,
                training_hooks = [logging_hook],
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.compat.v1.metrics.accuracy(
                labels = label, 
                predictions = tf.math.argmax(logits, axis=1))
            eval_loss = tf.compat.v1.metrics.mean(
                values=per_example_loss)
            
            #eval_metrics = (metric_fn, [per_sequence_loss_example, label_id, logits])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                scaffold=scaffold_fn,
                eval_metric_ops={
                    'accuracy':accuracy,
                    'eval_loss':eval_loss,
                    })
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                scaffold=scaffold_fn,
                predictions = {
                    'probabilities':probabilities
                    })
            
        return(output_spec)
    
    return(model_fn)

def label_str_to_numeric(resource_dir):
    label_map = {}
    buffer_df = pd.read_csv(os.path.join(resource_dir,'raw.tsv'), sep='\t')
    buffer_df = screen_for_language(buffer_df, resource_dir)
    label_list = sorted(set(buffer_df.loc[:,'label']))
    for key, val in enumerate(label_list):
        label_map.update({val:key})
    return(label_map)

def get_punc_pattern(pun_chars):
    regex_punc_pattern = '|'.join('\\'+str(i) for i in pun_chars)
    return(regex_punc_pattern)

def restore_id_mappings(resource_dir):
    buffer_df = pd.read_csv(os.path.join(resource_dir,'char_embed.tsv'), sep='\t')
    unigram_mapper = buffer_df#buffer_df.loc[buffer_df['feature'] == 'unigram']
    #bigram_mapper = buffer_df.loc[buffer_df['feature'] == 'bigram']
    return(unigram_mapper)

def inverse_map(label_map, predict_label):
    for (key, value) in six.iteritems(label_map):
        if value == int(predict_label):
            return key
    return('Not Listed')

def screen_for_language(buffer_data, resource_dir):
    
    label_df = pd.read_csv(os.path.join(resource_dir,'label.tsv'), sep='\t')
    temp_df = label_df.iloc[list(range(0,11)),:]#46,79#
    select_language = list(temp_df.loc[:,'label'])
    select_language = ['english','french','german','spanish','russian','italian','portugeese','hindi','irishgaelic','hebrew','slovak','danish','dutch','japanese']
    buffer_data = buffer_data.loc[buffer_data['label'].isin(select_language)]
    return(buffer_data)

def setup_cross_validation(resource_dir, n_fold):
    
    buffer_data = pd.read_csv(os.path.join(resource_dir,'raw.tsv'), sep='\t')
    buffer_data = screen_for_language(buffer_data, resource_dir)
    resource_folder = os.path.join(os.path.dirname(resource_dir),'resource')
    if not os.path.exists(resource_folder):
        os.mkdir(resource_folder)
    
    fold_wrapper = KFold(n_splits=n_fold)
    fold_index = 1
    start_range = buffer_data.shape[0]# 20000 #
    for range_index, test_index in fold_wrapper.split(range(start_range)):

        threshold = np.int(len(range_index)*0.02)
        eval_index = np.random.randint(
            low=range_index[0], high=range_index[-1],size=threshold)
        train_index = list(set(range_index).difference(set(eval_index)))
        #test_index = test_index[0:10]
            
        tf.compat.v1.logging.info("fold: %d" %fold_index+
                        "\t train: %d" %len(train_index)+
                        " eval: %d" %len(eval_index)+
                        " test: %d" %len(test_index))
        
        #cross fold validation folder
        curr_folder = os.path.join(resource_folder,str(fold_index))
        if not os.path.exists(curr_folder):
            os.mkdir(curr_folder)
        
        #train
        column = ['doc_id','encoding','sentence','label']
        train_df = buffer_data.iloc[train_index,:]
        train_df = train_df.loc[:,column]
        
        train_df.to_csv(os.path.join(curr_folder,'train.tsv'), sep='\t', index=False)
        
        #eval
        eval_df = buffer_data.iloc[eval_index,:]
        eval_df = eval_df.loc[:,column]
        eval_df.to_csv(os.path.join(curr_folder,'eval.tsv'), sep='\t', index=False)
          
        #test
        test_df = buffer_data.iloc[test_index,:]
        test_df = test_df.loc[:,column]
        test_df.to_csv(os.path.join(curr_folder,'test.tsv'), sep='\t', index=False)
        
        fold_index += 1

    return()

def get_index_delimiter(text_list, delim):
    split_index = []
    for item in text_list:
        if ord(item) == delim:
            split_index.append(text_list.index(item))
    return(split_index)

def split_sentences(text, regex_punc_pattern, seq_length):
    sentences = []
    sentences_length = []
    split_index = get_index_delimiter(list(text), ord('।'))
    if len(split_index) == 0:
        split_index.append(len(text))
    start = 0
    for end in split_index:
        sub_text = ''.join(i for i in list(text)[start:end+1])
        start = end+1 
        split_pattern = '(\. )|(\n)'
        for sentence in re.split(split_pattern, sub_text):
            if sentence is not None:
                sentence = str(sentence).lower().strip()
                sentence = re.sub(regex_punc_pattern, '', sentence)
                sentence = re.sub('\d+', '', sentence).strip()
                if len(sentence) > 1 and len(sentence) < seq_length:
                    sentences.append(sentence)
                    sentences_length.append(len(sentence))
    return(sentences, sentences_length)

def load_raw_file(resource_dir, regex_punc_pattern, seq_length, n_fold):
    
    buffer_df = None
    start_doc_id = 0
    doc_no = 1
    for curr_file in os.listdir(resource_dir):
        sfread = open(os.path.join(resource_dir, curr_file), 'r')
        s_currline = sfread.readline()
        while s_currline:
            sentences, sentences_length = split_sentences(s_currline, regex_punc_pattern, seq_length)
            if len(sentences) > 0:
                labels = list(['english']*len(sentences))
                encodings = list(['utf-8']*len(sentences))
                if buffer_df is not None:
                    start_doc_id = buffer_df.shape[0]
                doc_ids = []
                for i in range(len(sentences)):
                    start_doc_id += 1
                    doc_ids.append(str(curr_file)+'@'+str(i))
                
                print()
                local_df = pd.DataFrame(data=doc_ids, columns=['doc_id'])
                local_df.insert(loc=1, column='encoding',value=encodings)
                local_df.insert(loc=2, column='sentence',value=sentences)
                local_df.insert(loc=3, column='label',value=labels)
                #local_df.insert(loc=4, column='sentence_length',value=sentences_length)
                if buffer_df is None:
                    buffer_df = local_df
                else:
                    buffer_df = buffer_df.append(local_df, ignore_index=True)
            s_currline = sfread.readline()
            
        sfread.close()
        doc_no += 1
    
    input_dir = os.path.join(os.path.dirname(resource_dir),'resource')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    input_dir = os.path.join(input_dir,'predict')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    buffer_df.to_csv(os.path.join(input_dir,'test.tsv'), sep='\t', index=False)
    return()
    
def main(self):
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.get_logger().propagate = False
    
    #===========================================================================
    # process the raw input
    #===========================================================================
    
    sec_conversion = 1000000000
    start_time = time.perf_counter_ns()
    
    processor = InputProcessor()
    resource_dir = FLAGS.resource
    pun_chars = [';',':',',','.','!','?','(',')','[',']','#','@','$','"','\'','*']
    regex_punc_pattern = get_punc_pattern(pun_chars)
    n_fold = 1
    if FLAGS.eval_type == 'validation' and (
        FLAGS.do_train or FLAGS.do_eval or FLAGS.do_predict):
        n_fold = 5
        setup_cross_validation(resource_dir, n_fold)
        label_resource_dir = resource_dir
    elif FLAGS.eval_type == 'prediction' and FLAGS.do_predict:
        n_fold=1
        label_resource_dir = os.path.join(os.path.dirname(resource_dir),'nltk_corpora')
        load_raw_file(resource_dir, regex_punc_pattern, FLAGS.seq_length, n_fold)
    else:
        tf.compat.v1.logging.info('\n *****************'+'\n'+
                                  'EXECUTION TERMINATED \n Review eval_type and  do_train/eval/predict choice')
        sys.exit()
            
    #===========================================================================
    # loading label map : limits the number of languages detected by this model
    #===========================================================================
    label_map= label_str_to_numeric(label_resource_dir)
    print(label_map)
    unigram_mapper = restore_id_mappings(label_resource_dir)
    
    for i in range(1,(1+1)):
        tf.compat.v1.logging.info('*****************iteration no %d********************' %i)
        POINTER_RESOURCE_DIR = os.path.join(os.path.dirname(resource_dir),'resource',str(i))
        POINTER_OUTPUT_DIR = os.path.join(FLAGS.output,str(i))
        if FLAGS.eval_type == 'prediction':
            POINTER_RESOURCE_DIR = os.path.join(os.path.dirname(resource_dir),'resource','predict')
            POINTER_OUTPUT_DIR = os.path.join(FLAGS.output,'predict')
        if not os.path.exists(POINTER_OUTPUT_DIR):
            os.mkdir(POINTER_OUTPUT_DIR)
   
        #=======================================================================
        # define gpu server run config
        #=======================================================================
        tpu_cluster_resolver = None
        is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
        tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host)
        
        run_config = tf.compat.v1.estimator.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                master=FLAGS.master,
                model_dir = POINTER_OUTPUT_DIR,
                save_checkpoints_steps= FLAGS.save_checkpoints_steps,
                tpu_config=tpu_config)
        
        num_warmup_steps = None
        num_train_steps = None
        
        if FLAGS.do_train:
            tf.compat.v1.logging.info("***************Train***************")
            train_record_file = os.path.join(POINTER_OUTPUT_DIR,'train.tf_record')
            train_inputs = processor.get_train_data(POINTER_RESOURCE_DIR, FLAGS.seq_length)
            tf.compat.v1.logging.info('# of train input %d' %len(train_inputs))
            if not os.path.exists(train_record_file):
                map_input_to_features(train_inputs, FLAGS.seq_length,
                                      train_record_file, resource_dir,
                                      label_map, regex_punc_pattern,
                                      unigram_mapper)
                
            num_train_steps = int(
                            len(train_inputs)/FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
            train_input_fn = input_feature_builder(
                    input_file= train_record_file,
                    seq_length= FLAGS.seq_length,
                    is_training = True,
                    drop_remainder = True)
            
        model_fn = model_build(
            seq_length=FLAGS.seq_length,
            learning_rate=FLAGS.learning_rate,
            unigram_featuremap_size=unigram_mapper.shape[0], 
            symbol_group_size = len(symbol_group),
            hidden_dim=FLAGS.hidden_dim,
            num_label=len(label_map),
            num_train_steps = num_train_steps,
            num_warmup_steps = num_warmup_steps,
            init_checkpoint=FLAGS.init_checkpoint)
        
        estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu, 
            model_fn=model_fn, 
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)
            
        if FLAGS.do_train:  
            train_input_fn = input_feature_builder(
                    input_file= train_record_file,
                    seq_length= FLAGS.seq_length,
                    is_training = True,
                    drop_remainder = True)
            
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
            end_train_time = time.perf_counter_ns()
            cpu_execution_time_sec = (end_train_time - start_time)/sec_conversion
            tf.compat.v1.logging.info("Training time %s sec" %cpu_execution_time_sec)
            
        if FLAGS.do_eval:
            tf.compat.v1.logging.info("***************Eval***************")
            eval_record_file = os.path.join(POINTER_OUTPUT_DIR,'eval.tf_record')
            eval_input = processor.get_eval_data(POINTER_RESOURCE_DIR, FLAGS.seq_length)
            tf.compat.v1.logging.info('# of eval input %d' %len(eval_input))
            map_input_to_features(eval_input, FLAGS.seq_length,
                              eval_record_file, resource_dir,
                              label_map, regex_punc_pattern,
                              unigram_mapper)
            
            eval_steps=None
            if FLAGS.use_tpu:
                eval_steps = int(len(eval_input) // FLAGS.eval_batch_size)
            eval_drop_remainder = False
            eval_input_fn = input_feature_builder(
                input_file= eval_record_file,
                seq_length= FLAGS.seq_length,
                is_training = False,
                drop_remainder = eval_drop_remainder)
            
            result = estimator.evaluate(
                input_fn = eval_input_fn, 
                steps = eval_steps)
            
            for keys in result.keys():
                tf.compat.v1.logging.info("%s = %s" %(keys, str(result[keys])))
                
            end_eval_time = time.perf_counter_ns()
            cpu_execution_time_sec = (end_eval_time - start_time)/sec_conversion
            tf.compat.v1.logging.info("Eval time %s sec" %cpu_execution_time_sec)
                
        if FLAGS.do_predict:
            tf.compat.v1.logging.info("***************Predict***************")
            predict_record_file = os.path.join(POINTER_OUTPUT_DIR,'predict.tf_record')
            predict_input = processor.get_predict_data(POINTER_RESOURCE_DIR, FLAGS.seq_length)
            true_label = pd.read_csv(os.path.join(POINTER_RESOURCE_DIR,'test.tsv'),sep='\t')
            tf.compat.v1.logging.info('# of predict input %d' %len(predict_input))
            if not os.path.exists(predict_record_file):
                map_input_to_features(predict_input, FLAGS.seq_length,
                                  predict_record_file, resource_dir,
                                  label_map, regex_punc_pattern,
                                  unigram_mapper)
    
            predict_drop_remainder = False
            predict_input_fn = input_feature_builder(
                input_file= predict_record_file,
                seq_length= FLAGS.seq_length,
                is_training = False,
                drop_remainder = predict_drop_remainder)
            
            result = estimator.predict(input_fn=predict_input_fn)
            doc_set = {}
            with tf.io.gfile.GFile(os.path.join(POINTER_OUTPUT_DIR,'predicted_'+str(i)+'.tsv'), 'w+') as writer_buffer:
                writer_buffer.write('index\tpredict\ttrue\tsentence\n') 
                for (index, prediction) in enumerate(result):
                    probabilities = prediction['probabilities']
                    predict_id = np.argmax(probabilities)
                    #print(probabilities,'\t',predict_id)
                    predict_label = inverse_map(label_map, predict_id)
                    writer_buffer.write(str(true_label.loc[index,'doc_id'])+'\t'+str(predict_label)
                                        +'\t'+str(true_label.loc[index,'label'])
                                        +'\t'+str(true_label.loc[index,'sentence'])+'\n')
                    doc_name = true_label.loc[index,'doc_id'].split('@')[0]
                    doc_prect_val = []
                    if doc_name in doc_set:
                        doc_prect_val = doc_set.get(doc_name)
                    doc_prect_val.append(predict_label)
                    doc_set.update({doc_name:doc_prect_val})
                    
            writer_buffer.close()
            
            with tf.io.gfile.GFile(os.path.join(POINTER_OUTPUT_DIR,'predict_summary'+str(i)+'.tsv'), 'w+') as writer_buffer:
                writer_buffer.write('doc_name\tpredict_language\n') 
                for key, val in six.iteritems(doc_set):
                    value_str = ', '.join(str(i) for i in list(set(val)))
                    writer_buffer.write(str(key)+'\t'+str(value_str)+'\n') 
            writer_buffer.close()
            
if __name__ == '__main__':
    
    flags.mark_flag_as_required('type')
    flags.mark_flag_as_required('eval_type')
    tf.compat.v1.app.run()
    