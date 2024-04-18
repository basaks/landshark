FROM python:3.9-slim
MAINTAINER Sudipta Basak <sudipta@tasnix.com>

WORKDIR /usr/src/landshark

RUN apt update && apt upgrade -y
RUN apt-get install -y --no-install-recommends \
        make \
        gcc \
        libc6-dev \
        libopenblas-dev \
        libgdal-dev  \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/* \
    && alias pip=pip3

RUN pip install -U pip
# RUN pip install -e .[dev]
