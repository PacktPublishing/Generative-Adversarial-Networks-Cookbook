FROM base_image

RUN git clone https://github.com/fyu/lsun.git

RUN cd lsun && python2.7 download.py -c church_outdoor

RUN apt install -y unzip
# RUN unzip church_outdoor_train_lmdb.zip
RUN cd lsun && unzip church_outdoor_val_lmdb.zip
RUN cd lsun && mkdir data && python data.py export *_val_lmdb --out_dir data
