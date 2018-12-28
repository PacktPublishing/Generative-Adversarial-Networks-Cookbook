#/bin/bash

# Training Step
xhost +
docker run -it \
   --runtime=nvidia \
   --rm \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $HOME/Chapter5/data:/data \
   -v $HOME/Chapter5/src:/src \
   ch5 python3 /src/run.py
