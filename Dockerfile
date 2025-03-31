FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 기본 패키지 및 Python 3.10 관련 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# python 명령어를 python3.10으로 지정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# pip 업그레이드
RUN python -m pip install --upgrade pip

# PyTorch와 관련 라이브러리 설치 (CUDA 11.7 지원)
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

CMD [ "python" ]
