params_model='model_GD360_h184_w184_c6_3conv_L0.7_O1.0_U1.0_7_0.71-6381.meta'
search_dir='./tensor_model_sum'
n_hidden=512
do_active_fields=0
thresh=0
#must set multiple test on

###########################################################################
for model in "$search_dir"/*
#model name output format: ./testbench/6classes/2/model_GD3conv360_h184_w184_c63conv_O0.75_U1.0_1_0.79-431.meta
do
  
  if [[ $model != *"meta"* ]] && [[ $model != *checkpoint* ]];then
    #split string by '/'
    file_0="$(echo $model | rev | cut -d/ -f1 | rev)"    
    acc_0="$(echo $model | rev | cut -d_ -f1 | rev)"
    acc="$(echo $acc_0 | cut -d- -f1)"
    echo $acc,$model
    file="eva-$file_0.txt"
    #==,>,<
    if (( $(echo "$acc>=$thresh" | bc -l) ));then
      echo 'save eva file' $file,$model
      python ./src/tensor_eva.py $model $n_hidden $do_active_fields > $file
      #python ./src/tensor_eva.py $model $n_hidden $do_active_fields
    
    #else
      #echo 'out of threshold bound'
    fi
    
  fi

done



#########################################################################
search_dir2='./logs/1'
#search_dir2='./testbench/6classes/1'
thresh2=0.5
###########################################################################
for model in "$search_dir2"/*
do
  #echo $model
  if [[ $model != *"meta"* ]] && [[ $model != *checkpoint* ]];then
    acc_0="$(echo $model | rev | cut -d_ -f1 | rev)"
    echo $fn,$model
    acc="$(echo $acc_0 | cut -d- -f1)"
    
    
    if (( $(echo "$acc<=$thresh2" | bc -l) ));then
      echo 'remove' $acc_0,$model
      rm $model*
        
    fi
    
  fi

done
