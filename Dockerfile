FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# (선택사항) 필요한 패키지 설치 예시
# RUN apt-get update && apt-get install -y <필요한_패키지들> && rm -rf /var/lib/apt/lists/*

# 기본 CMD 설정
CMD ["python"]
