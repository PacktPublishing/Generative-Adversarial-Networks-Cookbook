#/bin/bash

# Training Step
xhost +
docker run -it \
   --runtime=nvidia \
   --rm \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $HOME/simGAN/out:/out \
   -v $HOME/simGAN/src:/src \
   ch7 python3 /src/run.py
