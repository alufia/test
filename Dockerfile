FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# deadsnakes PPA 추가 및 필요한 패키지 설치 (curl 포함)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl

# deadsnakes PPA 추가 (Python 3.12 제공)
RUN add-apt-repository ppa:deadsnakes/ppa

# Python 3.12 및 관련 패키지 설치 (pip 설치를 위해 distutils 포함)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# 기본 python 명령어를 python3.12로 지정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# get-pip.py로 pip 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# pip 업그레이드 (선택사항)
RUN python -m pip install --upgrade pip

# PyTorch 설치 (현재 CUDA 12.x용 PyTorch 휠은 cu121이 제공됨)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

CMD ["python"]
