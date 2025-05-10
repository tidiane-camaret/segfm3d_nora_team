FROM python:3.13-slim-bookworm

WORKDIR /workspace

## Install dependencies
# RUN pip install numpy torch scikit-image connected-components-3d

## Copy dependencies
# external dependencies
COPY nnInteractive /workspace/nnInteractive
RUN pip install -e /workspace/nnInteractive

# local depedencies
RUN mkdir -p /workspace/src
COPY method.py /workspace/src/method.py

## Copy model and scripts
COPY model/ /workspace/model/
COPY predict.sh /workspace/predict.sh
COPY predict.py /workspace/predict.py

# Make the predict.sh executable
RUN chmod +x /workspace/predict.sh

# Set correct permissions for input/output directories
RUN mkdir -p /workspace/inputs /workspace/outputs