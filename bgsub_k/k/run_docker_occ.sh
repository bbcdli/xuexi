docker stop occupancy
docker rm -f occupancy
docker run --name occupancy -u `id -u` -it -v /home/hy/.Xauthority:/home/hy/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -v /etc/localtime:/etc/localtime:ro -e DISPLAY -v `pwd`:/home/occupancy/ -p 8008:8008 -w /home/occupancy/ docker_occ /bin/bash

#if use -u to add user, USER must be set when building the Image already

#sudo apt-get install libcanberra-gtk-module:i386   -u `id -u`
#-v /etc/localtime:/etc/localtime:ro 

#docker run --name occupancy -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v `pwd`:/home/occupancy/ -p 8008:8008 -w /home/occupancy/ docker_occ /bin/bash python tmp.py

#docker run -it --name occupancy -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/hy/Documents/hy_dev/occupancy/:/home/occupancy/ -p 8008:8008 -w /home/occupancy/ docker_occ /bin/bash

#docker run --name occupancy -it -v /home/hy/.Xauthority:/home/hy/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -v `pwd`:/home/occupancy/ -p 8008:8008 -w /home/occupancy/ docker_occ /bin/bash

