FROM base_image

RUN mkdir data && cd data && wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
RUN cd data && wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

RUN pip install ipython==5

ADD preprocess.py /
