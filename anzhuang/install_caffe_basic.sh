#sudo apt-get install python-dev -y 
#sudo apt-get install python-pip -y 
#sudo apt-get install python-opencv -y 
#sudo apt-get install libhdf5-dev -y 
#sudo apt-get install python-h5py -y 
#sudo apt-get install ffmpeg -y 
#pip install tensorflow==1.2 
#pip install Pillow 
#pip install keras==2.0.0 
#pip install matplotlib
#
#sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler -y
#sudo apt-get install --no-install-recommends libboost-all-dev -y
#sudo apt-get install libatlas-base-dev -y
#sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev -y
#mkdir .local/install
#cd .local/install
#git clone https://github.com/BVLC/caffe.git
#cd ./caffe/python
#for req in $(cat requirements.txt); do pip install $req; done
##for req in $(cat requirements.txt); do pip install $req --user; done #if error,then add --user
#check site package availablity $python -m site
#.local/install/caffe/$cp Makefile.config.example Makefile.config
#-------------------------------------------
# set nano Makefile.config
#CPU_ONLY
#PYTHON_INCLUDE := /usr/include/python2.7 \
#     /home/hy/.local/lib/python2.7/site-packages/numpy/core/include
#INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
#LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/
#-------------------------------------------
#.local/install/caffe/$make all
#.local/install/caffe/$make clean
#.local/install/caffe/$make all
#.local/install/caffe/$make runtest
#.local/install/caffe$make pycaffe
#.local/install/caffe/$pip install pydot
#.local/install/caffe/$sudo apt-get install dot  #//can also without it once
#.local/install/caffe/$sudo apt-get install graphviz
#.local/install/caffe$make pytest
#.local/install/caffe$export PYTHONPATH='/home/hy/.local/install/caffe/python'
