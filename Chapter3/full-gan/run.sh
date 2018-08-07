#/bin/bash
nvidia-docker build -t ch3 . 

xhost +
docker run -it \
   --runtime=nvidia \
   --rm \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $HOME/full-gan/data:/data \
   ch3 python3 run.py
