#/bin/bash
xhost +

docker run -it \
   --runtime=nvidia \
   --rm \
   -v $HOME/book_repos/Chapter5/data:/data \
   ch5 wget -N https://raw.githubusercontent.com/junyanz/CycleGAN/master/datasets/download_dataset.sh -O /data/download_dataset.sh  && ./download_dataset.sh horse2zebra
