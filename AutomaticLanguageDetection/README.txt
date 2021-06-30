# Automatic Language Detection 
# V: 0.0.1
# Author: Neha Warikoo
# Dated : 29th June 2021

# INFO: This python-based module is developed for detecting language of a document.
# It uses a composite feature (Word vector-embedding + Linguistic feature) on a 
# Convolution Neural Network(CNN)+Dense Layer model to train nltk/udhr based dataset for 
# language detection task. 

# Training Data:
# For the purposes of this demo, the language classification has been limited to below mentioned classes
# The constraint was added to adjust for data limitation and training duration
# In its present capacity the model is trained to detect below set of 14 languages: 
# ['english','french','german','spanish','russian','italian','portugeese','hindi','irishgaelic','hebrew','slovak','danish','dutch','japanese']
# The scope of language detection can be increased with additional training data.
# We have attempted to provide a mix of similar dialect languages as well as the scripts using
# different character sets for training. With this mixed combination, we attempt to demonstrate
# the effectiveness of our approach in classifying text based on local phrase context 
# (for similar script languages e.g. german v dutch) as well as based on dissimilar tokens
# (for dissimilar script languages e.g. english vs japanese)   
# Use below configuration for training:
$PYTHON_PATH/python $MYPATH/AutomaticLanguageDetection/src/com/prj/bundle/executable/main_autodetection.py \
	--do_train=True \
	--do_eval=False \
	--do_predict=False \
	--eval_type=validation \
	--type=sentence \
	--seq_length=1000 \
	--hidden_dim=768 \
	--learning_rate=1e-5 \
	--num_train_epochs=15 \
	--init_checkpoint='' \
	--resource=$MYPATH'/AutomaticLanguageDetection/src/com/prj/bundle/nltk_corpora' \
	--output=$MYPATH'/AutomaticLanguageDetection/src/com/prj/bundle/output'

# Execution details:
# The prediction scope is limited to the 
# The program can be executed from command line using the shell script 'run_autodetect_server.sh'
# to predict language for unknown dataset, use below configuration:

$PYTHON_PATH/python $MYPATH/AutomaticLanguageDetection/src/com/prj/bundle/executable/main_autodetection.py \
	--do_train=False \
	--do_eval=False \
	--do_predict=True \
	--eval_type=prediction \
	--type=sentence \
	--seq_length=1000 \
	--hidden_dim=768 \
	--learning_rate=1e-5 \
	--num_train_epochs=15 \
	--init_checkpoint=$MYPATH'/AutomaticLanguageDetection/src/com/prj/bundle/output/1/model.ckpt-3416' \
	--resource=$MYPATH'/AutomaticLanguageDetection/src/com/prj/bundle/input' \
	--output=$MYPATH'/AutomaticLanguageDetection/src/com/prj/bundle/output'
	
# The prediction results are stored $MYPATH'/AutomaticLanguageDetection/src/com/prj/bundle/output/predict' folder
# Document wise prediction file : predict_summary1.tsv
# Senetence level prediction file : predicted_1.tsv
# Set MYPATH = executable path on local/server_cluster
   