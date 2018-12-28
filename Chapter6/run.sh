#/bin/bash

# Training Step
xhost +
docker run -it \
   --runtime=nvidia \
   --rm \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $HOME/Chapter6/out:/out \
   -v $HOME/Chapter6/src:/src \
   ch6 python3 /src/run.py
