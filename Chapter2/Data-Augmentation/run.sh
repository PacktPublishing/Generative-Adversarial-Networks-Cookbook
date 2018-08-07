#/bin/bash
nvidia-docker build -t ch2 . 

xhost +
docker run -it \
   --runtime=nvidia \
   --rm \
   -e DISPLAY=$DISPLAY \
   -e SCIPY_PIL_IMAGE_VIEWER=/usr/bin/eog \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   ch2 python aug_demo.py
