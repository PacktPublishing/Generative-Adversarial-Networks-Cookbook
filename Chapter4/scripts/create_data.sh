#/bin/bash
xhost +

# Download the data into our data folder
docker run -it \
   --runtime=nvidia \
   --rm \
   -v $HOME/DCGAN/data:/data \
   ch4 python lsun/download.py -o /data  -c church_outdoor && unzip church_outdoor_train_lmdb.zip && unzip church_outdoor_val_lmdb.zip && mkdir /data/church_outdoor_train_lmdb/expanded

# Expand the data into our data folder
docker run -it \
   --runtime=nvidia \
   --rm \
   -v $HOME/DCGAN/data:/data \
   ch4 python lsun/data.py export /data/church_outdoor_train_lmdb --out_dir /data/church_outdoor_train_lmdb/expanded --flat 


# Save to NPY File
docker run -it \
   --runtime=nvidia \
   --rm \
   -v $HOME/DCGAN/data:/data \
   -v $HOME/Chapter4/DCGAN/src:/src \
   ch4 python3 src/save_to_npy.py
