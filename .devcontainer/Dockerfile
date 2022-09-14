FROM registry.hub.docker.com/nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG PYTHON_VERSION="3.10.7"
ARG PYTHON_PATH=/usr/local/python
ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=/usr/local/py-utils/bin
ENV PATH=${PYTHON_PATH}/bin:${PATH}:${PIPX_BIN_DIR}

COPY library-scripts/*.sh /tmp/library-scripts/

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive  \
    && /bin/bash /tmp/library-scripts/common-debian.sh \
    && apt-get install -y openmpi-bin libopenmpi-dev \
    && /bin/bash /tmp/library-scripts/python-debian.sh "${PYTHON_VERSION}" "${PYTHON_PATH}" "${PIPX_HOME}" "automatic" "true" "false" "true" "true" \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/library-scripts
