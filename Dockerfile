FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git build-essential libssl-dev unzip aria2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda and create Python 3.10 environment
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -f /tmp/miniconda.sh && \
    /opt/conda/bin/conda create -y -n py310 python=3.10 pip && \
    /opt/conda/bin/conda clean -afy

# Activate Python 3.10 environment by default for interactive shells
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate py310" >> /etc/bash.bashrc

ENV PATH="/opt/conda/envs/py310/bin:/opt/conda/bin:$PATH" \
    SHELL="/bin/bash"

CMD ["bash"]