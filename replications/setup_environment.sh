#!/bin/bash
# =============================================================================
# setup_environment.sh
# Run this script once inside WSL (Ubuntu) to create the conda environment.
#
# Usage:
#   chmod +x replications/setup_environment.sh
#   bash replications/setup_environment.sh
# =============================================================================

set -e

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="dr_env"

# ── Install Miniconda if not already installed ────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# ── Create environment ────────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.12..."
    conda create -y -n "$ENV_NAME" python=3.12
fi

conda activate "$ENV_NAME"

# ── Install packages ──────────────────────────────────────────────────────────
echo "Installing PyTorch (CPU)..."
pip install --quiet torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

echo "Installing core dependencies..."
pip install --quiet \
    accelerate==1.5.2 \
    anthropic==0.42.0 \
    datasets==3.1.0 \
    numpy==2.1.3 \
    openai==1.59.6 \
    peft==0.15.0 \
    transformers==4.50.0 \
    trl==0.16.0 \
    google-generativeai==0.8.3 \
    sentencepiece \
    "tokenizers>=0.21,<0.22" \
    protobuf \
    litellm \
    huggingface_hub

echo "Installing llama-cpp-python (CPU server)..."
CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_METAL=OFF" pip install --quiet "llama-cpp-python[server]"

echo ""
echo "Done! Activate the environment with: conda activate dr_env"
