FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install to /opt instead of /workspace to avoid volume conflicts
WORKDIR /opt

# Clean up and install git
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone ai-toolkit repository to /opt
RUN git clone https://github.com/ostris/ai-toolkit.git /opt/ai-toolkit

# Change to ai-toolkit directory and update submodules
WORKDIR /opt/ai-toolkit
RUN git submodule update --init --recursive

# Create virtual environment that inherits system packages
RUN python -m venv venv --system-site-packages

# Upgrade pip
RUN /opt/ai-toolkit/venv/bin/pip install --upgrade pip

# Install ONLY the lightweight packages
RUN /opt/ai-toolkit/venv/bin/pip install --no-cache-dir \
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

# Install the git packages separately
RUN /opt/ai-toolkit/venv/bin/pip install --no-cache-dir \
    git+https://github.com/jaretburkett/easy_dwpose.git \
    git+https://github.com/huggingface/diffusers@363d1ab7e24c5ed6c190abb00df66d9edb74383b

# Create config.yaml file
RUN printf 'job: extension\nconfig:\n  name: "hd_sh_v5"\n  process:\n    - type: '\''sd_trainer'\''\n      training_folder: "output"\n      device: cuda:0\n      trigger_word: "th3k"\n      network:\n        type: "lora"\n        linear: 32\n        linear_alpha: 32\n        network_kwargs:\n          ignore_if_contains:\n            - "ff_i.experts"\n            - "ff_i.gate"\n      save:\n        dtype: bfloat16\n        save_every: 250\n        max_step_saves_to_keep: 10\n      datasets:\n        - folder_path: "/workspace/ai-toolkit/dataset"\n          caption_ext: "txt"\n          caption_dropout_rate: 0.05\n          resolution: [ 1024 ]\n      train:\n        batch_size: 1\n        steps: 1500\n        gradient_accumulation_steps: 1\n        train_unet: true\n        train_text_encoder: false\n        gradient_checkpointing: true\n        noise_scheduler: "flowmatch"\n        timestep_type: shift\n        optimizer: "adamw8bit"\n        lr: 2e-4\n        ema_config:\n          use_ema: false\n          ema_decay: 0.99\n        dtype: bf16\n      model:\n        name_or_path: "HiDream-ai/HiDream-I1-Full"\n        extras_name_or_path: "HiDream-ai/HiDream-I1-Full"\n        arch: "hidream"\n        quantize: true\n        quantize_te: true\n        model_kwargs:\n          llama_model_path: "unsloth/Meta-Llama-3.1-8B-Instruct"\n      sample:\n        sampler: "flowmatch"\n        sample_every: 250\n        width: 1024\n        height: 1024\n        prompts:\n          - "a th3k sketch of a pig"\n          - "a th3k sketch of a ballerina "\n          - "a th3k sketch of a lamborghini"\n          - "a th3k sketch of a dog dancing with a zebra"\n          - "a th3k sketch of a kangaroo on a motorcycle"\n        neg: ""\n        seed: 42\n        walk_seed: true\n        guidance_scale: 4\n        sample_steps: 25\nmeta:\n  name: "[name]"\n  version: '\''5.0'\''' > /opt/ai-toolkit/config.yaml

# Configure Jupyter to work without password
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py

# Add venv activation to bashrc
RUN echo 'if [ -d "/workspace/ai-toolkit/venv" ]; then' >> /root/.bashrc && \
    echo '    source /workspace/ai-toolkit/venv/bin/activate' >> /root/.bashrc && \
    echo '    echo "✅ ai-toolkit virtual environment activated!"' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc

# Create a simple copy script that runs once
RUN printf '#!/bin/bash\nif [ ! -d "/workspace/ai-toolkit" ]; then\n    echo "Copying ai-toolkit to workspace..."\n    cp -r /opt/ai-toolkit /workspace/\n    echo "Done! ai-toolkit is ready in /workspace/ai-toolkit"\nfi\n' > /opt/copy-to-workspace.sh && chmod +x /opt/copy-to-workspace.sh

# Create a script that runs the copy and then calls original start.sh
RUN printf '#!/bin/bash\n# Copy ai-toolkit if needed\n/opt/copy-to-workspace.sh\n\n# Call the original RunPod start script\nexec /start.sh "$@"\n' > /opt/runpod-start.sh && chmod +x /opt/runpod-start.sh

# Use our script as the entrypoint
ENTRYPOINT ["/opt/runpod-start.sh"]

# Back to workspace
WORKDIR /workspace
