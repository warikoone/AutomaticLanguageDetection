#$ -e /vol/projects/nwarikoo/AutomaticLanguageDetection/src/com/prj/bundle/executable/
#$ -o /vol/projects/nwarikoo/AutomaticLanguageDetection/src/com/prj/bundle/executable/
#$ -cwd
#$ -l arch=linux-x64   
#$ -pe multislot 4
MYPATH='/vol/projects/nwarikoo'
export MYPATH='/vol/projects/nwarikoo'
echo $PATH
echo $MYPATH

/home/nwarikoo/anaconda3/bin/python $MYPATH/AutomaticLanguageDetection/src/com/prj/bundle/executable/main_autodetection.py \
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
