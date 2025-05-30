FROM python:3.13-slim-bookworm

WORKDIR /workspace

ARG PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

## Copy dependencies
# external dependencies
RUN pip install monai nnunetv2

# local depedencies
RUN mkdir -p /workspace/src
COPY src /workspace/src

## Copy model and scripts
COPY model/ /workspace/model/
COPY predict.sh /workspace/predict.sh
COPY predict.py /workspace/predict.py

RUN chmod +x /workspace/predict.sh && mkdir -p /workspace/inputs /workspace/outputs && apt-get clean && rm -rf /var/lib/apt/lists/*
