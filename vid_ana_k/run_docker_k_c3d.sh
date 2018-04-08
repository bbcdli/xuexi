#env installation before caffe
#apt-get update
#apt-get install python-dev
#apt-get install python-opencv
#apt-get install python-pip -y
#apt-get install libhdf5-dev -y
#apt-get install python-h5py -y
#apt-get install ffmpeg -y
#pip install tensorflow
#pip install Pillow
#pip install keras
#pip install matplotlib

#run docker for keras c3d, installed full env on Image 'mymod/aggrcaff:v1.1'
#without caffe can also run train_k.py
docker stop caffev1_1
docker rm -f caffev1_1
#docker run -it --name caffev1_1 -v /${PWD}/:/home/ -w /home/ mymod/aggrcaff:v1.1 /bin/bash

#without x11,can run train
#docker run --name caffev1_1 -u `id -u` -it -v /home/hy/.Xauthority:/home/hy/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -v /etc/localtime:/etc/localtime:ro -e DISPLAY -v `pwd`:/home/ -p 8008:8008 -w /home/ mymod/aggrcaff:v1.1 /bin/bash

#with x11,can also run eva, so -v aggr_vids is required
docker run --name caffev1_1 -it -v /home/hy/.Xauthority:/home/hy/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -v `pwd`:/home/c3d_keras/ -v `pwd`/../aggr_vids/:/home/aggr_vids/ -p 8008:8008 -w /home/c3d_keras/ mymod/aggrcaff:v1.1 /bin/bash

docker start caffev1_1
docker attach caffev1_1

#run docker for tensorflow c3d
#docker run --name aggr -it -v /home/hy/Documents/hy_dev/aggr/:/home/hy_dev/ -v /home/hy/Documents/hy_dev/aggr/aggr_vids/:/home/aggr_vids/ -w /home/hy_dev/ u4_aggrtf1
