FROM base_image
RUN apt update && apt install -y python3-pydot python-pydot-ng graphviz
RUN pip3 install ipython
RUN pip3 uninstall keras -y && pip3 install keras==2.2.1

# Download and Install the Keras Contributor layers
# Some papers have layers not implemented in default keras
RUN git clone https://www.github.com/keras-team/keras-contrib.git
RUN cd keras-contrib && git checkout 04b64a47a7552f && python setup.py install
RUN cd keras-contrib && python3 setup.py install