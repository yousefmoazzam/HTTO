FROM registry.hub.docker.com/nvidia/cuda:11.7.1-runtime-ubuntu20.04

ENV HTTO_DIR=/htto
WORKDIR ${HTTO_DIR}

ARG PYTHON_VERSION="3.10.7"

RUN apt-get update -y \
    && apt-get install -y \
        build-essential \
        libssl-dev \
        libncurses5-dev \
        libsqlite3-dev \
        libreadline-dev \
        libtk8.6 \
        libgdm-dev \
        libdb4o-cil-dev \
        libpcap-dev \
        wget \
    && cd /tmp \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --with-ensurepip --enable-optimizations --with-lto --enable-profiling \
    && make install -j

RUN apt-get update -y \
    && apt-get install -y \
        openmpi-bin \
        libopenmpi-dev \
        libhdf5-openmpi-dev

COPY . ${HTTO_DIR}

RUN pip3 install . -r requirements.txt

ENTRYPOINT ["python3.10", "-m", "htto"]