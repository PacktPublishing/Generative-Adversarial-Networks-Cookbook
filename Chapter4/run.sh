#/bin/bash

# Training Step
xhost +
docker run -it \
   --runtime=nvidia \
   --rm \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /home/jk/Desktop/book_repos/Chapter4/DCGAN/data:/data \
   -v /home/jk/Desktop/book_repos/Chapter4/DCGAN/src:/src \
   ch4 python3 /src/run.py
