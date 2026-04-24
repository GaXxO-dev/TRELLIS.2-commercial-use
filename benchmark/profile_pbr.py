#!/usr/bin/env python
"""
Profile PBR rendering to identify bottlenecks.
"""
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(str(PROJECT_DIR))

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap

IMAGE_PATH = "assets/example_image/T.png"

class CudaTimer:
    def __init__(self):
        self.start_event = None
        self.end_event = None
        self.elapsed_ms = 0
    
    def start(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self
    
    def stop(self):
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        return self.elapsed_ms

def profile_ssao():
    """Profile SSAO computation in isolation."""
    print("\n" + "=" * 80)
    print("PROFILING SSAO")
    print("=" * 80)
    
    from trellis2.renderers.pbr_mesh_renderer import screen_space_ambient_occlusion
    
    H, W = 512, 512
    device = 'cuda'
    
    depth = torch.rand(H, W, 1, device=device) * 10 + 5
    normal = torch.randn(H, W, 3, device=device)
    normal = torch.nn.functional.normalize(normal, dim=-1)
    
    perspective = torch.eye(4, device=device)
    perspective[0, 0] = 500
    perspective[1, 1] = 500
    perspective[0, 2] = 0
    perspective[1, 2] = 0
    
    timer = CudaTimer()
    
    for samples in [16, 32, 64, 128]:
        torch.cuda.synchronize()
        timer.start()
        f_occ = screen_space_ambient_occlusion(depth, normal, perspective, samples=samples)
        elapsed = timer.stop()
        print(f"  samples={samples}: {elapsed:.1f} ms")

def profile_cubemap_sampling():
    """Profile cubemap sampling."""
    print("\n" + "=" * 80)
    print("PROFILING CUBEMAP SAMPLING")
    print("=" * 80)
    
    from trellis2.renderers.pbr_envmap import sample_cubemap, sample_cubemap_mip, build_cubemap_mips
    
    device = 'cuda'
    
    cubemap = torch.rand(6, 512, 512, 3, device=device)
    directions = torch.randn(1024, 1024, 3, device=device)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    
    timer = CudaTimer()
    
    torch.cuda.synchronize()
    timer.start()
    result = sample_cubemap(cubemap, directions)
    elapsed = timer.stop()
    print(f"  sample_cubemap (1024x1024): {elapsed:.1f} ms")
    
    mips = build_cubemap_mips(cubemap)
    mip_level = torch.ones(1024, 1024, device=device) * 2.5
    
    torch.cuda.synchronize()
    timer.start()
    result = sample_cubemap_mip(mips, directions, mip_level)
    elapsed = timer.stop()
    print(f"  sample_cubemap_mip (1024x1024): {elapsed:.1f} ms")

def profile_depth_peeling():
    """Profile depth peeling layers."""
    print("\n" + "=" * 80)
    print("PROFILING DEPTH PEELING")
    print("=" * 80)
    
    from trellis2.utils.drtk_compat import RasterizeCudaContext, DepthPeeler
    from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection
    import utils3d
    
    device = 'cuda'
    resolution = 1024
    
    num_vertices = 100000
    vertices = torch.randn(num_vertices, 3, device=device) * 0.5
    faces = torch.randint(0, num_vertices, (num_vertices // 3, 3), device=device, dtype=torch.int32)
    
    timer = CudaTimer()
    glctx = RasterizeCudaContext(device=device)
    
    extrinsics = torch.eye(4, device=device)
    extrinsics[2, 3] = 3
    intrinsics = torch.eye(3, device=device)
    intrinsics[0, 0] = 500
    intrinsics[1, 1] = 500
    
    perspective = intrinsics_to_projection(intrinsics, 0.1, 100.0)
    full_proj = perspective @ extrinsics
    
    vertices_homo = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=-1)
    vertices_clip = (vertices_homo @ full_proj.T).unsqueeze(0)
    
    times_per_layer = []
    
    torch.cuda.synchronize()
    total_start = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    with DepthPeeler(glctx, vertices_clip, faces, (resolution, resolution)) as peeler:
        for layer in range(8):
            torch.cuda.synchronize()
            layer_start = torch.cuda.Event(enable_timing=True)
            layer_start.record()
            
            rast, rast_db = peeler.rasterize_next_layer()
            
            torch.cuda.synchronize()
            layer_end = torch.cuda.Event(enable_timing=True)
            layer_end.record()
            torch.cuda.synchronize()
            layer_time = layer_start.elapsed_time(layer_end)
            times_per_layer.append(layer_time)
    
    total_end = torch.cuda.Event(enable_timing=True)
    total_end.record()
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    print(f"  Total depth peeling (8 layers): {total_time:.1f} ms")
    for i, t in enumerate(times_per_layer):
        print(f"    Layer {i}: {t:.1f} ms")

def main():
    print("=" * 80)
    print("PBR RENDERING PROFILER")
    print("=" * 80)
    
    profile_ssao()
    profile_cubemap_sampling()
    profile_depth_peeling()
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()