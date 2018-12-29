FROM base_image

# Installations for graphing and analysis
RUN apt update && apt install -y python3-pydot python-pydot-ng graphviz
RUN pip3 install kaggle ipython pillow

# Copy Kaggle.json
COPY kaggle.json /root/.kaggle/kaggle.json

# Download the Data
RUN kaggle datasets download -d daavoo/3d-mnist
RUN unzip 3d-mnist.zip -d 3d-mnist
RUN rm 3d-mnist.zip
WORKDIR /src
