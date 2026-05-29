FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_RETRIES=10
ENV PIP_TIMEOUT=300

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir --timeout=300 \
    torch==2.5.1+cu124 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir --ignore-installed \
    numpy pandas scikit-learn matplotlib \
    jupyter transformers datasets accelerate safetensors flask

WORKDIR /workspace
VOLUME ["/workspace", "/models"]
EXPOSE 8888
CMD ["python"]
