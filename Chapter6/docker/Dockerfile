FROM base_image
RUN apt update && apt install -y python3-pydot python-pydot-ng graphviz
RUN pip3 install ipython
RUN pip3 uninstall keras -y && pip3 install keras==2.2.1

RUN wget -N http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz
RUN mkdir -p /data/cityscapes/
RUN tar -zxvf cityscapes.tar.gz -C /data/cityscapes/
RUN rm cityscapes.tar.gz

