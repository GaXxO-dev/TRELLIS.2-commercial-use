# TRELLIS.2 Project Notes

## Project Overview
3D generative model (4B parameters) for image-to-3D generation using O-Voxel sparse representation.

## System Requirements
- **Linux only** (verified on Linux, not tested elsewhere)
- **NVIDIA GPU**: 24GB+ VRAM required (verified on A100/H100)
- **CUDA Toolkit**: 12.4 recommended. Set `CUDA_HOME` if multiple versions installed.
- **Python**: 3.8+

## Installation

Full setup (new conda env):
```sh
. ./setup.sh --new-env --basic --flash-attn --drtk --nvdiffrec --cumesh --o-voxel --flexgemm
```

Creates conda environment `trellis2` with PyTorch 2.6.0 + CUDA 12.4.

For V100 or GPUs without flash-attn support: install `xformers` manually, then set `ATTN_BACKEND=xformers`.

## Entry Points
- `train.py` — Training script (distributed GPU supported via `--num_gpus`, `--num_nodes`)
- `app.py` — Gradio demo for image-to-3D
- `app_texturing.py` — Gradio demo for PBR texturing
- `example.py` / `example_texturing.py` — Minimal inference examples

## Inference Requirements
Scripts must set environment variables **before imports**:
```python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

Pipeline loading:
```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
```

## Architecture
- `trellis2/` — Core package (models, pipelines, trainers, datasets, renderers)
- `o-voxel/` — Separate package for sparse voxel operations (installed via setup.sh)
- `data_toolkit/` — Dataset preparation scripts (see README there for pipeline)

## Training Configs
JSON configs in `configs/scvae/` (VAE) and `configs/gen/` (flow models).

Example:
```sh
python train.py --config configs/scvae/shape_vae_next_dc_f16c32_fp16.json --output_dir results/shape_vae --data_dir "{...}"
```

Dataset paths passed as JSON strings. See README for full training pipeline.

## Dependencies with Separate Licenses
- `DRTK` — Differentiable rendering toolkit (MIT license)
- `nvdiffrec` — PBR split-sum renderer (own license)

## No Tests
No test suite or CI present in this repo.