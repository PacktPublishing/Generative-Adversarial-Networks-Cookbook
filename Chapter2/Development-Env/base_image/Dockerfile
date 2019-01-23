FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ARG KERAS=2.2.0
ARG TENSORFLOW=1.8.0

# Update the repositories within the container
RUN apt-get update

# Install Python 2 and 3 + our basic dev tools
RUN apt-get install -y \
          python-dev \
          python3-dev \
          curl \
          git \
          vim \
          python-pip \
          python-opencv \
          python3-pip \
          python-tk \ 
          python3-tk \
          wget \
          unzip

# Install Tensorflow and Keras for Python 2
RUN pip --no-cache-dir install \
         tensorflow_gpu==${TENSORFLOW} \ 
         keras==${KERAS} \
         numpy \
         scipy \
         lmdb \ 
         matplotlib==2.2.3 \ 
         pandas \
         pillow
         

# Install Tensorflow and Keras for Python 3
RUN pip3 --no-cache-dir install \
         tensorflow_gpu==${TENSORFLOW} \ 
         keras==${KERAS} \
         numpy \
         scipy \
         lmdb \
         matplotlib \
         pandas \
         pillow
