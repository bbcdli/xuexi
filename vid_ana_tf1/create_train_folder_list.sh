#/bin/bash
#create test/train list
SUB_FOLDER=-1
LABEL_0=0
LABEL_1=1
LABEL_2=2
LABEL_3=3
USE_ORI_SCRIPT=0
LIST_TYPE='test_clipfolder'
#./create_train_list.sh /home/hy/Documents/aggr/Own_Data/images  #no / at the end
WORK_DIR=$(pwd)
images_root_folder=$WORK_DIR'/Own_Data/images'
OUT_FILE=$WORK_DIR'/lists'/$LIST_TYPE'.list'
> $OUT_FILE

if [[ $USE_ORI_SCRIPT == "0" ]]; then
  #for folder in $1/*
  for folder in $images_root_folder/*
  do
    echo $folder
    for images_cl_folder in "$folder"/*
    do
      if [[ $folder == *"0"* ]];then
 echo "$images_cl_folder" $LABEL_0 >> $OUT_FILE
      fi
      if [[ $folder == *"1"* ]];then
        echo "$images_cl_folder" $LABEL_1 >> $OUT_FILE
      fi
      if [[ $folder == *"2"* ]];then
        echo "$images_cl_folder" $LABEL_2 >> $OUT_FILE
      fi
      if [[ $folder == *"3"* ]];then
        echo "$images_cl_folder" $LABEL_3 >> $OUT_FILE
      fi
    done
  done
fi


LIST_TYPE='train_clipfolder'
OUT_FILE=$WORK_DIR'/lists'/$LIST_TYPE'.list'
> $OUT_FILE
if [[ $USE_ORI_SCRIPT == "0" ]]; then
  #for folder in $1/*
  for folder in $images_root_folder/*
  do
    echo $folder
    for images_cl_folder in "$folder"/*
    do
      if [[ $folder == *"0"* ]];then
 echo "$images_cl_folder" $LABEL_0 >> $OUT_FILE
      fi
      if [[ $folder == *"1"* ]];then
        echo "$images_cl_folder" $LABEL_1 >> $OUT_FILE
      fi
      if [[ $folder == *"2"* ]];then
        echo "$images_cl_folder" $LABEL_2 >> $OUT_FILE
      fi
      if [[ $folder == *"3"* ]];then
        echo "$images_cl_folder" $LABEL_3 >> $OUT_FILE
      fi
    done
  done
fi

