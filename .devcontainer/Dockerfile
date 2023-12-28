FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID


# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt update \
    && apt install -y sudo curl \
    && mkdir -p /etc/sudoers.d \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN groupadd -g 1005 vglusers && \
    usermod -a -G video,sudo $USERNAME

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && sh ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

USER $USERNAME

ENV PATH /opt/conda/bin:$PATH

RUN conda init

RUN echo "conda activate" >> ~/.bashrc