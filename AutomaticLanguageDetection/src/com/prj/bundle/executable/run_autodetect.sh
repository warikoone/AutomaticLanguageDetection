
MYPATH='/home/epid/Disk_R/P_Workspace/Workspace_ALD_1.0/AutomaticLanguageDetection'
export MYPATH

python main_autodetection.py \
	--do_train=True \
	--do_eval=False \
	--do_predict=False \
	--type=sentence \
	--seq_length=1000 \
	--hidden_dim=768 \
	--learning_rate=1e-3 \
	--num_train_epochs=3 \
	--init_checkpoint='' \
	--resource='/home/epid/Disk_R/P_Workspace/Workspace_ALD_1.0/AutomaticLanguageDetection/src/com/prj/bundle/nltk_corpora' \
	--output='/home/epid/Disk_R/P_Workspace/Workspace_ALD_1.0/AutomaticLanguageDetection/src/com/prj/bundle/output'