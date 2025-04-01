# NVIDIA CUDA 기반 이미지 사용 (Ubuntu 22.04, CUDA 12.3)
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# 패키지 설치 시 인터랙티브 프롬프트 방지
ENV DEBIAN_FRONTEND=noninteractive

# PPA 추가 및 소프트웨어 속성 관리 도구 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12을 설치하기 위해 deadsnakes PPA 추가
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Python 3.12 및 관련 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12을 기본 Python 인터프리터로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# pip 최신 버전으로 업그레이드
RUN python -m pip install --upgrade pip

# CUDA 12.3 지원과 함께 PyTorch 설치
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123

# 컨테이너 실행 시 기본적으로 Python 실행
CMD ["python"]
