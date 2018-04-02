#run_app.sh
PROJ_DIR=${PWD} #'../occupancy/'
SAVE_LOG_PATH=$PROJ_DIR/testbench/
train_mode=new_train  #con_train
model_path=$PROJ_DIR/testbench/1/
model_path_name=''
data_path=$PROJ_DIR/Data/ #Data/training/2A/
test_data_path=$PROJ_DIR/Test_Data/
log_LABEL='1'
learning_rate=0.000098
batch_size=1
MAX_ITERATION=3
INPUT_SIZE=160
rnd_blurriness_min=150
rnd_blurriness_max=280
rnd_darkness_min=29
rnd_darkness_max=150
dropouts='[0.15,0.25,0.34,0.5,1,0.4,0.25,0.15,0.15]'
#
echo $PROJ_DIR
SCRIPT=train_ssh_args.py
python $SCRIPT $PROJ_DIR $train_mode $model_path $model_path_name $data_path  $test_data_path $log_LABEL $learning_rate  $batch_size $MAX_ITERATION $INPUT_SIZE $rnd_blurriness_min $rnd_blurriness_max $rnd_darkness_min $rnd_darkness_max $dropouts

