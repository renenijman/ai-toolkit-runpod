FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Clean up and install git
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone ai-toolkit repository
RUN git clone https://github.com/ostris/ai-toolkit.git

# Change to ai-toolkit directory and update submodules
WORKDIR /workspace/ai-toolkit
RUN git submodule update --init --recursive

# Create virtual environment that inherits system packages
RUN python -m venv venv --system-site-packages

# Upgrade pip
RUN /workspace/ai-toolkit/venv/bin/pip install --upgrade pip

# Install ONLY the lightweight packages (skip torch, torchvision, torchao)
RUN /workspace/ai-toolkit/venv/bin/pip install --no-cache-dir \
    safetensors \
    transformers==4.52.4 \
    lycoris-lora==1.8.3 \
    flatten_json \
    pyyaml \
    oyaml \
    tensorboard \
    kornia \
    invisible-watermark \
    einops \
    accelerate \
    toml \
    pydantic \
    omegaconf \
    k-diffusion \
    open_clip_torch \
    timm \
    prodigyopt \
    python-dotenv \
    bitsandbytes \
    hf_transfer \
    lpips \
    pytorch_fid \
    sentencepiece \
    huggingface_hub \
    peft \
    gradio \
    python-slugify \
    opencv-python \
    matplotlib==3.10.1

# Install the git packages separately (these might need the others installed first)
RUN /workspace/ai-toolkit/venv/bin/pip install --no-cache-dir \
    git+https://github.com/jaretburkett/easy_dwpose.git \
    git+https://github.com/huggingface/diffusers@363d1ab7e24c5ed6c190abb00df66d9edb74383b

# Back to workspace
WORKDIR /workspace
