FROM continuumio/miniconda3 as conda_upstream

RUN groupadd -r conda --gid 900 \
    && chown -R :conda /opt/conda \
    && chmod -R g+w /opt/conda \
    && find /opt -type d | xargs -n 1 chmod g+s

FROM registry.hub.docker.com/nvidia/cuda:11.7.1-devel-ubuntu20.04
COPY --from=conda_upstream /opt /opt/

COPY .devcontainer/library-scripts/* /tmp/library-scripts/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
ARG USERNAME=vscode
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && /bin/bash /tmp/library-scripts/common-debian.sh \
    && mv -f "/tmp/library-scripts/meta.env" /usr/local/etc/vscode-dev-containers/meta.env \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate htto" >> ~/.bashrc \
    && groupadd -r conda --gid 900 \
    && usermod -aG conda ${USERNAME} \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/add-notice.sh

ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=/usr/local/py-utils/bin
ENV PATH=${PATH}:${PIPX_BIN_DIR}
RUN bash /tmp/library-scripts/python-debian.sh "none" "/opt/conda" "${PIPX_HOME}" "${USERNAME}" "true" \ 
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* .devcontainer/library-scripts/python-debian.sh

COPY conda/environment_explicit.txt /tmp/conda-env/
RUN umask 0002 \
    && /opt/conda/bin/conda create -n htto --file /tmp/conda-env/environment_explicit.txt --no-default-packages \
    && rm -rf /tmp/conda-env
