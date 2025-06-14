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

# Create virtual environment WITHOUT system packages (FIXED)
RUN python -m venv venv

# IMPORTANT: Use the venv python for all installations
ENV PATH="/opt/ai-toolkit/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install ALL packages from requirements.txt (FIXED)
RUN pip install --no-cache-dir -r requirements.txt

# Install the git packages separately
RUN pip install --no-cache-dir \
    git+https://github.com/jaretburkett/easy_dwpose.git \
    git+https://github.com/huggingface/diffusers@363d1ab7e24c5ed6c190abb00df66d9edb74383b

# Install huggingface_hub for model downloading
RUN pip install --no-cache-dir huggingface_hub

# Set up Hugging Face cache directory
ENV HF_HOME=/opt/huggingface_cache
ENV TRANSFORMERS_CACHE=/opt/huggingface_cache
ENV HF_DATASETS_CACHE=/opt/huggingface_cache

# Create cache directory
RUN mkdir -p /opt/huggingface_cache

# Copy the model download script
COPY download_models.py /opt/download_models.py

# Run the model download and verification script
RUN python /opt/download_models.py

# Verify critical packages are installed
RUN python -c "import torch; import transformers; import diffusers; print('✅ All packages successfully installed')"

# Create config.yaml file
RUN printf 'job: extension\nconfig:\n  name: "hd_sh_v5"\n  process:\n    - type: '\''sd_trainer'\''\n      training_folder: "output"\n      device: cuda:0\n      trigger_word: "th3k"\n      network:\n        type: "lora"\n        linear: 32\n        linear_alpha: 32\n        network_kwargs:\n          ignore_if_contains:\n            - "ff_i.experts"\n            - "ff_i.gate"\n      save:\n        dtype: bfloat16\n        save_every: 250\n        max_step_saves_to_keep: 10\n      datasets:\n        - folder_path: "/workspace/ai-toolkit/dataset"\n          caption_ext: "txt"\n          caption_dropout_rate: 0.05\n          resolution: [ 1024 ]\n      train:\n        batch_size: 1\n        steps: 1500\n        gradient_accumulation_steps: 1\n        train_unet: true\n        train_text_encoder: false\n        gradient_checkpointing: true\n        noise_scheduler: "flowmatch"\n        timestep_type: shift\n        optimizer: "adamw8bit"\n        lr: 2e-4\n        ema_config:\n          use_ema: false\n          ema_decay: 0.99\n        dtype: bf16\n      model:\n        name_or_path: "HiDream-ai/HiDream-I1-Full"\n        extras_name_or_path: "HiDream-ai/HiDream-I1-Full"\n        arch: "hidream"\n        quantize: true\n        quantize_te: true\n        model_kwargs:\n          llama_model_path: "unsloth/Meta-Llama-3.1-8B-Instruct"\n      sample:\n        sampler: "flowmatch"\n        sample_every: 250\n        width: 1024\n        height: 1024\n        prompts:\n          - "a th3k sketch of a pig"\n          - "a th3k sketch of a ballerina "\n          - "a th3k sketch of a lamborghini"\n          - "a th3k sketch of a dog dancing with a zebra"\n          - "a th3k sketch of a kangaroo on a motorcycle"\n        neg: ""\n        seed: 42\n        walk_seed: true\n        guidance_scale: 4\n        sample_steps: 25\nmeta:\n  name: "[name]"\n  version: '\''5.0'\''' > /opt/ai-toolkit/config.yaml

# Configure Jupyter to work without password
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py

# Add venv activation to bashrc with proper PATH and HF cache
RUN echo 'if [ -d "/workspace/ai-toolkit/venv" ]; then' >> /root/.bashrc && \
    echo '    export PATH="/workspace/ai-toolkit/venv/bin:$PATH"' >> /root/.bashrc && \
    echo '    export HF_HOME=/opt/huggingface_cache' >> /root/.bashrc && \
    echo '    export TRANSFORMERS_CACHE=/opt/huggingface_cache' >> /root/.bashrc && \
    echo '    export HF_DATASETS_CACHE=/opt/huggingface_cache' >> /root/.bashrc && \
    echo '    source /workspace/ai-toolkit/venv/bin/activate' >> /root/.bashrc && \
    echo '    echo "✅ ai-toolkit virtual environment activated!"' >> /root/.bashrc && \
    echo '    echo "💡 Use: python run.py config.yaml"' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc

# Create a simple copy script that runs once
RUN printf '#!/bin/bash\nif [ ! -d "/workspace/ai-toolkit" ]; then\n    echo "Copying ai-toolkit to workspace..."\n    cp -r /opt/ai-toolkit /workspace/\n    echo "Done! ai-toolkit is ready in /workspace/ai-toolkit"\nfi\n\n# Set up HF cache environment for runtime\nexport HF_HOME=/opt/huggingface_cache\nexport TRANSFORMERS_CACHE=/opt/huggingface_cache\nexport HF_DATASETS_CACHE=/opt/huggingface_cache\n' > /opt/copy-to-workspace.sh && chmod +x /opt/copy-to-workspace.sh

# Create a script that runs the copy, starts without-password Jupyter, then calls original start.sh
RUN printf '#!/bin/bash\n# Copy ai-toolkit if needed\n/opt/copy-to-workspace.sh\n\n# Remove any Jupyter token environment variables\nunset JUPYTER_TOKEN\nunset JUPYTER_PASSWORD\n\n# Start a background process to fix Jupyter after RunPod starts it\n(\n  sleep 5\n  echo "Restarting Jupyter without password..."\n  pkill -f jupyter-lab\n  sleep 2\n  nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token="" --ServerApp.password="" --ServerApp.allow_origin="*" --ServerApp.base_url=/ > /tmp/jupyter.log 2>&1 &\n  echo "Jupyter Lab started without password!"\n) &\n\n# Call the original RunPod start script\nexec /start.sh "$@"\n' > /opt/runpod-start.sh && chmod +x /opt/runpod-start.sh

# Use our script as the entrypoint
ENTRYPOINT ["/opt/runpod-start.sh"]

# Back to workspace
WORKDIR /workspace