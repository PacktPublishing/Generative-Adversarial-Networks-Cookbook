#/bin/bash
xhost +

docker run -it \
   --runtime=nvidia \
   --rm \
   -v $HOME/Chapter6/data:/data \
   ch6 wget -N https://raw.githubusercontent.com/phillipi/pix2pix/master/datasets/download_dataset.sh -O /data/download_dataset.sh  &&   bash -E "/data/download_dataset.sh maps"
