#!/usr/bin/env fish
# TRELLIS.2 Setup Script for Fish Shell
# Usage: source setup.fish [OPTIONS]
#
# IMPORTANT: Before running, ensure CUDA_HOME is set to CUDA 12.4:
#   export CUDA_HOME="$HOME/.local/cuda-12.4"  # or /usr/local/cuda-12.4
#   export PATH="$CUDA_HOME/bin:$PATH"
#   export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

set -l HELP false
set -l NEW_ENV false
set -l BASIC false
set -l FLASHATTN false
set -l CUMESH false
set -l OVOXEL false
set -l FLEXGEMM false
set -l DRTK false

# Parse arguments
for arg in $argv
    switch $arg
        case '-h' '--help'
            set HELP true
        case '--new-env'
            set NEW_ENV true
        case '--basic'
            set BASIC true
        case '--flash-attn'
            set FLASHATTN true
        case '--cumesh'
            set CUMESH true
        case '--o-voxel'
            set OVOXEL true
        case '--flexgemm'
            set FLEXGEMM true
        case '--drtk'
            set DRTK true
        case '*'
            echo "Error: Invalid argument '$arg'"
            set HELP true
    end
end

if test (count $argv) -eq 0
    set HELP true
end

if test "$HELP" = "true"
    echo "Usage: source setup.fish [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --flash-attn            Install flash-attention"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --drtk                  Install DRTK (differentiable renderer, MIT license)"
    return 0
end

# Get system information
set -l WORKDIR (pwd)
set -l PLATFORM ""

if command -v nvidia-smi > /dev/null
    set PLATFORM "cuda"
else if command -v rocminfo > /dev/null
    set PLATFORM "hip"
else
    echo "Error: No supported GPU found"
    return 1
end

# Detect distro for package manager
set -l DISTRO ""
if test -f /etc/debian_version
    set DISTRO "debian"
else if test -f /etc/arch-release
    set DISTRO "arch"
else if test -f /etc/fedora-release
    set DISTRO "fedora"
end

# Check CUDA_HOME
if test -z "$CUDA_HOME"
    echo "WARNING: CUDA_HOME is not set. CUDA extensions require CUDA 12.4."
    echo "Set CUDA_HOME before running this script:"
    echo '  export CUDA_HOME="$HOME/.local/cuda-12.4"'
    echo '  export PATH="$CUDA_HOME/bin:$PATH"'
    echo '  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"'
end

# Initialize git submodules (required for o-voxel's eigen dependency)
echo "[INIT] Initializing git submodules..."
git submodule update --init --recursive

# Create new conda environment
if test "$NEW_ENV" = "true"
    echo "[NEW_ENV] Creating conda environment 'trellis2' with Python 3.10..."
    conda create -n trellis2 python=3.10 -y
    conda activate trellis2
    
    if test "$PLATFORM" = "cuda"
        echo "[NEW_ENV] Installing PyTorch 2.6.0 with CUDA 12.4..."
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    else if test "$PLATFORM" = "hip"
        echo "[NEW_ENV] Installing PyTorch 2.6.0 with ROCm 6.2.4..."
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    end
end

# Install basic dependencies
if test "$BASIC" = "true"
    echo "[BASIC] Installing basic Python dependencies..."
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard 'pygltflib>=1.16.0'
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    
    echo "[BASIC] Installing libjpeg development library..."
    switch $DISTRO
        case "debian"
            sudo apt install -y libjpeg-dev
        case "arch"
            sudo pacman -S --noconfirm libjpeg-turbo
        case "fedora"
            sudo dnf install -y libjpeg-turbo-devel
        case '*'
            echo "[BASIC] Warning: Could not detect distro. Install libjpeg manually if needed."
    end
    
    echo "[BASIC] Installing pillow-simd, kornia, timm, psutil..."
    pip install pillow-simd kornia timm psutil
end

# Install flash-attention
if test "$FLASHATTN" = "true"
    if test "$PLATFORM" = "cuda"
        echo "[FLASHATTN] Installing flash-attn 2.7.3 (prebuilt wheel for PyTorch 2.6 + CUDA 12.4)..."
        # Prebuilt wheel is faster and more reliable than building from source
        pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    else if test "$PLATFORM" = "hip"
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        env GPU_ARCHS=gfx942 python setup.py install
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    end
end

# Install DRTK (requires patches for CPU kernels and setuptools compatibility)
if test "$DRTK" = "true"
    echo "[DRTK] Installing DRTK with patches for CPU kernel support..."
    
    set -l ORIG_SETUPTOOLS (pip show setuptools 2>/dev/null | grep -oP '^Version: \K.*' || echo "82.0.1")
    echo "[DRTK] Temporarily downgrading setuptools (required for pkg_resources)..."
    pip install setuptools==69.5.1
    
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/DRTK
    git clone https://github.com/facebookresearch/DRTK.git /tmp/extensions/DRTK
    
    # Patch setup.py to include missing CPU kernel sources
    # 1. Add interpolate_kernel_cpu.cpp
    sed -i 's|"src/interpolate/interpolate_kernel.cu",|"src/interpolate/interpolate_kernel.cu",\n                    "src/interpolate/interpolate_kernel_cpu.cpp",|' /tmp/extensions/DRTK/setup.py
    
    # 2. Add rasterize_kernel_cpu.cpp
    sed -i 's|"src/rasterize/rasterize_kernel.cu",|"src/rasterize/rasterize_kernel.cu",\n                    "src/rasterize/rasterize_kernel_cpu.cpp",|' /tmp/extensions/DRTK/setup.py
    
    # 3. Add edge_grad_kernel_cpu.cpp
    sed -i 's|"src/edge_grad/edge_grad_kernel.cu",|"src/edge_grad/edge_grad_kernel.cu",\n                    "src/edge_grad/edge_grad_kernel_cpu.cpp",|' /tmp/extensions/DRTK/setup.py
    
    # 4. Add render_kernel_cpu.cpp
    sed -i 's|"src/render/render_kernel.cu", "src/render/render_module.cpp"|"src/render/render_kernel.cu", "src/render/render_module.cpp", "src/render/render_kernel_cpu.cpp"|' /tmp/extensions/DRTK/setup.py
    
    # 5. Patch cpu_atomic.h for C++17 compatibility (use reference instead of copy)
    sed -i 's/auto target = detail::atomic_ref_at/auto\& target = detail::atomic_ref_at/g' /tmp/extensions/DRTK/src/include/cpu_atomic.h
    
    echo "[DRTK] Building DRTK (this may take a few minutes)..."
    pip install /tmp/extensions/DRTK --no-build-isolation
    
    echo "[DRTK] Restoring setuptools to $ORIG_SETUPTOOLS..."
    pip install setuptools==$ORIG_SETUPTOOLS
end

# Install CuMesh
if test "$CUMESH" = "true"
    echo "[CUMESH] Installing CuMesh..."
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    pip install /tmp/extensions/CuMesh --no-build-isolation
end

# Install FlexGEMM
if test "$FLEXGEMM" = "true"
    echo "[FLEXGEMM] Installing FlexGEMM..."
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    pip install /tmp/extensions/FlexGEMM --no-build-isolation
end

# Install o-voxel (editable mode to find trellis2/utils)
if test "$OVOXEL" = "true"
    echo "[OVOXEL] Installing o-voxel in editable mode (requires project directory)..."
    pip install -e "$WORKDIR/o-voxel" --no-build-isolation
end

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "  conda activate trellis2"
echo ""
echo "To verify installation:"
echo "  cd $WORKDIR"
echo "  python -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')\""
echo "  python -c \"import flash_attn; print(f'flash-attn {flash_attn.__version__}')\""
echo "  python -c \"import drtk; print('drtk OK')\""
echo "  PYTHONPATH=. python -c \"import o_voxel; print('o-voxel OK')\""