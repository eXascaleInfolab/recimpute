# base image with Ubuntu 22.04
FROM ubuntu:22.04

# set non-interactive mode for apt-get to prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# install basic utilities
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    gnupg \
    git \
    sudo \
    curl \
    nano \
    build-essential \
    cmake \
    libopenmpi-dev \
    libopenblas-dev \
    liblapack-dev \
    libarmadillo-dev \
    libmlpack-dev \
    octave \
    liboctave-dev \
    r-base \
    gnuplot \
    python2-dev \
    python2 \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# add the Mono key and repository
RUN gpg --homedir /tmp --no-default-keyring --keyring /usr/share/keyrings/mono-official-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF && \
    echo "deb [signed-by=/usr/share/keyrings/mono-official-archive-keyring.gpg] https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list

# update package lists and install Mono
RUN apt-get update && \
    apt-get install -y mono-devel && \
    rm -rf /var/lib/apt/lists/*

# clone the ImputeBench repository
WORKDIR /app
RUN git clone https://github.com/eXascaleInfolab/bench-vldb20 imputebench

# install ImputeBench
WORKDIR /app/imputebench
RUN chmod +x install_linux.sh && \
    sh install_linux.sh

# save ImputeBench path
WORKDIR /app/imputebench/TestingFramework/bin/Debug
RUN echo $(pwd) > /tmp/benchmark_path

# clone RecImpute and checkout to the correct branch
WORKDIR /app
RUN git clone https://github.com/eXascaleInfolab/recimpute.git
WORKDIR /app/recimpute
ARG RECIMPUTE_BRANCH=revisions
RUN git checkout ${RECIMPUTE_BRANCH}

# update RecImpute config with the correct ImputeBench path
RUN sed -i "s|BENCHMARK_PATH: .*|BENCHMARK_PATH: $(cat /tmp/benchmark_path)|" Config/imputebenchlabeler_config.yaml

# install RecImpute
RUN chmod +x install_script.sh && \
    sh install_script.sh

CMD ["/bin/bash"]

