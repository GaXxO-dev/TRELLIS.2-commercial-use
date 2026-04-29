# TRELLIS.2 Docker Environment Specification (Inference Only)
# For exact environment replication

# =============================================================================
# SYSTEM REQUIREMENTS
# =============================================================================
# OS: Linux (tested on Ubuntu 22.04)
# Python: 3.10.x (3.10.20 verified)
# CUDA Toolkit: 12.4.x (12.4.131 verified)
# NVIDIA Driver: >= 525.x (595.58.03 verified)
# GPU: NVIDIA with 24GB+ VRAM (RTX 4090 / A100 / H100 verified)
# Conda: Any recent version (25.x+)

# =============================================================================
# ENVIRONMENT VARIABLES (set before imports)
# =============================================================================
# OPENCV_IO_ENABLE_OPENEXR=1
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# STEP 1: Create conda environment
# =============================================================================
# conda create -n trellis2 python=3.10 -y
# conda activate trellis2

# =============================================================================
# STEP 2: Install PyTorch (CUDA 12.4)
# =============================================================================
# pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# =============================================================================
# STEP 3: Install core dependencies
# =============================================================================
# pip install -r requirements-inference.txt

# =============================================================================
# STEP 4: Install git dependencies
# =============================================================================
# pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# =============================================================================
# STEP 5: Install prebuilt wheels
# =============================================================================
# flash-attention
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#
# DRTK (prebuilt for CUDA 12.4, Python 3.10)
# pip install https://github.com/GaXxO-dev/TRELLIS.2-commercial-use/releases/download/v0.1.0/drtk-0.1.0+cuda124-cp310-cp310-linux_x86_64.whl

# =============================================================================
# STEP 6: Build CUDA extensions (requires CUDA_HOME set)
# =============================================================================
# export CUDA_HOME=/usr/local/cuda-12.4  # or $HOME/.local/cuda-12.4
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Option A: Use setup.sh/setup.fish
#   source setup.sh --flash-attn --drtk --cumesh --o-voxel --flexgemm
#
# Option B: Build manually (see requirements-inference.txt for commands)

# =============================================================================
# STEP 7: Verify installation
# =============================================================================
# python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')"
# python -c "import drtk; print('drtk OK')"
# python -c "import o_voxel; print('o-voxel OK')"
# python -c "import cumesh; print('cumesh OK')"
# python -c "import flex_gemm; print('flex_gemm OK')"
# PYTHONPATH=. python -c "from trellis2.pipelines import Trellis2ImageTo3DPipeline; print('TRELLIS pipeline OK')"

# =============================================================================
# DOCKERFILE TEMPLATE
# =============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    git libjpeg-dev libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Environment variables
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

WORKDIR /app

# Copy requirements first for caching
COPY requirements-inference.txt /app/

# Install PyTorch
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
RUN pip install --no-cache-dir -r requirements-inference.txt

# Install git dependency
RUN pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install prebuilt wheels
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir https://github.com/GaXxO-dev/TRELLIS.2-commercial-use/releases/download/v0.1.0/drtk-0.1.0+cuda124-cp310-cp310-linux_x86_64.whl

# Copy project (for o-voxel)
COPY . /app/

# Build CUDA extensions (CuMesh, FlexGEMM, o-voxel)
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive && \
    pip install --no-cache-dir /tmp/CuMesh --no-build-isolation

RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive && \
    pip install --no-cache-dir /tmp/FlexGEMM --no-build-isolation

RUN pip install -e /app/o-voxel --no-build-isolation

# Clean up
RUN rm -rf /tmp/CuMesh /tmp/FlexGEMM /root/.cache/pip

# Verify
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')" && \
    python -c "import drtk; print('drtk OK')"

# Default command
CMD ["python", "example.py"]