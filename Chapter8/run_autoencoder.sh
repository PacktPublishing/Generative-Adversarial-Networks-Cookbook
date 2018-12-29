#/bin/bash

# Run autoencoder step
xhost +
docker run -it \
   --runtime=nvidia \
   --rm \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $HOME/3d-gan-from-images/out:/out \
   -v $HOME/3d-gan-from-images/src:/src \
   ch8 python3 /src/encoder.py
