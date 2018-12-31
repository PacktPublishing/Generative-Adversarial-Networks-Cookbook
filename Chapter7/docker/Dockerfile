FROM base_image

# Installations for graphing and analysis
RUN apt update && apt install -y python3-pydot python-pydot-ng graphviz
RUN pip3 install kaggle ipython pillow

# Copy Kaggle.json
COPY kaggle.json ~/.kaggle/kaggle.json

# Download the Data
RUN kaggle datasets download -d 4quant/eye-gaze -p /data/
WORKDIR /data
RUN unzip eye-gaze.zip -d eye-gaze
RUN rm eye-gaze.zip
WORKDIR /src

