#!/usr/bin/env python
"""
Inference Speed Benchmark for Fork (DRTK) Implementation.
Full pipeline: image preprocessing → GLB export
Resolution: 512 | PBR preview: 8 views, 1024 | Texture: 1024
"""
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

import sys
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(str(PROJECT_DIR))

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils, glb_utils
from trellis2.renderers import EnvMap

# Configuration
SEED = 42
IMAGE_PATH = "assets/example_image/T.png"
RESOLUTION = "512"
DECIMATION_TARGET = 500000
TEXTURE_SIZE = 1024
PBR_VIEWS = 8
PBR_RESOLUTION = 1024
NUM_ITERATIONS = 5
OUTPUT_DIR = SCRIPT_DIR

# Timing utilities
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

def get_memory_usage():
    return torch.cuda.max_memory_allocated() / 1024**3

def reset_memory_tracking():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

def run_benchmark():
    print("=" * 80)
    print("INFERENCE SPEED BENCHMARK: Fork (DRTK)")
    print("=" * 80)
    print(f"Resolution: {RESOLUTION}")
    print(f"PBR Preview: {PBR_VIEWS} views @ {PBR_RESOLUTION}px")
    print(f"Texture Size: {TEXTURE_SIZE}px")
    print(f"Iterations: {NUM_ITERATIONS} (1 warmup + {NUM_ITERATIONS-1} measured)")
    print(f"Image: {IMAGE_PATH}")
    print()
    
    # Load environment map for PBR rendering
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))
    
    # Load pipeline (NOT TIMED)
    print("Loading pipeline (not timed)...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print("Pipeline loaded.")
    print()
    
    # Load image once (NOT TIMED)
    image_raw = Image.open(IMAGE_PATH)
    
    results = []
    
    for iteration in range(NUM_ITERATIONS):
        iter_name = "warmup" if iteration == 0 else f"iter_{iteration}"
        print(f"--- {iter_name.upper()} ---")
        
        reset_memory_tracking()
        timings = {}
        timer = CudaTimer()
        
        # Step 1: Image Preprocessing
        torch.cuda.synchronize()
        timer.start()
        image = pipeline.preprocess_image(image_raw)
        timings['preprocessing'] = timer.stop()
        
        # Step 2: Conditioning Extraction
        torch.cuda.synchronize()
        timer.start()
        torch.manual_seed(SEED)
        cond_512 = pipeline.get_cond([image], 512)
        cond_1024 = pipeline.get_cond([image], 1024)
        timings['cond_extraction'] = timer.stop()
        
        # Step 3: Sparse Structure Sampling
        torch.cuda.synchronize()
        timer.start()
        ss_res = 32
        coords = pipeline.sample_sparse_structure(cond_512, ss_res, 1, {})
        timings['sparse_sampling'] = timer.stop()
        
        # Step 4: Shape SLat Sampling
        torch.cuda.synchronize()
        timer.start()
        shape_slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models['shape_slat_flow_model_512'],
            coords, {}
        )
        timings['shape_slat_sampling'] = timer.stop()
        
        # Step 5: Texture SLat Sampling
        torch.cuda.synchronize()
        timer.start()
        tex_slat = pipeline.sample_tex_slat(
            cond_512, pipeline.models['tex_slat_flow_model_512'],
            shape_slat, {}
        )
        timings['tex_slat_sampling'] = timer.stop()
        
        # Step 6: Latent Decoding
        torch.cuda.synchronize()
        timer.start()
        torch.cuda.empty_cache()
        mesh = pipeline.decode_latent(shape_slat, tex_slat, 512)[0]
        timings['latent_decoding'] = timer.stop()
        
        # Step 7: Mesh Simplification
        torch.cuda.synchronize()
        timer.start()
        mesh.simplify(16777216)
        timings['mesh_simplify'] = timer.stop()
        
        # Step 8: PBR Preview Rendering (render_snapshot)
        torch.cuda.synchronize()
        timer.start()
        pbr_images = render_utils.render_snapshot(
            mesh, resolution=PBR_RESOLUTION, r=2, fov=36, 
            nviews=PBR_VIEWS, envmap=envmap
        )
        timings['pbr_render'] = timer.stop()
        
        # Step 9-12: GLB Export (timed internally by o_voxel)
        torch.cuda.synchronize()
        o_voxel.postprocess.enable_benchmark()
        timer.start()
        glb_mesh = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=512,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=DECIMATION_TARGET,
            texture_size=TEXTURE_SIZE,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        timings['glb_export'] = timer.stop()
        
        # Get sub-timings from o_voxel
        glb_timings = o_voxel.postprocess.get_benchmark_timings()
        o_voxel.postprocess.disable_benchmark()
        
        timings['glb_fill_holes'] = glb_timings.get('glb_fill_holes', 0)
        timings['glb_bvh'] = glb_timings.get('glb_bvh', 0)
        timings['glb_remesh'] = glb_timings.get('glb_remesh', 0)
        timings['glb_uv_unwrap'] = glb_timings.get('glb_uv_unwrap', 0)
        timings['glb_texture_bake'] = glb_timings.get('glb_texture_bake', 0)
        timings['glb_finalize'] = glb_timings.get('glb_finalize', 0)
        
        # Calculate total
        timings['total'] = sum([
            timings['preprocessing'],
            timings['cond_extraction'],
            timings['sparse_sampling'],
            timings['shape_slat_sampling'],
            timings['tex_slat_sampling'],
            timings['latent_decoding'],
            timings['mesh_simplify'],
            timings['pbr_render'],
            timings['glb_export'],
        ])
        
        # Memory usage
        timings['peak_memory_gb'] = get_memory_usage()
        
        print(f"  Total: {timings['total']:.1f} ms ({timings['total']/1000:.2f}s)")
        print(f"  Memory: {timings['peak_memory_gb']:.2f} GB")
        
        results.append({
            'iteration': iter_name,
            'timings': timings
        })
        
        torch.cuda.empty_cache()
    
    # Calculate averages (excluding warmup)
    avg_timings = {}
    measured_results = results[1:]
    
    for key in measured_results[0]['timings'].keys():
        if key == 'peak_memory_gb':
            avg_timings[key] = max(r['timings'][key] for r in measured_results)
        else:
            avg_timings[key] = sum(r['timings'][key] for r in measured_results) / len(measured_results)
    
    return {
        'config': {
            'resolution': RESOLUTION,
            'pbr_views': PBR_VIEWS,
            'pbr_resolution': PBR_RESOLUTION,
            'texture_size': TEXTURE_SIZE,
            'decimation_target': DECIMATION_TARGET,
            'iterations': NUM_ITERATIONS,
        },
        'warmup': results[0]['timings'],
        'average': avg_timings,
        'iterations': [{k: v for k, v in r['timings'].items() if k != 'peak_memory_gb'} for r in measured_results],
    }

def print_results(data):
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS: Fork (DRTK)")
    print("=" * 80)
    
    # Warmup
    print()
    print("WARMUP (First Iteration)")
    print("-" * 80)
    warmup = data['warmup']
    print(f"{'Step':<35} {'Time (ms)':<12}")
    print("-" * 80)
    for key in ['preprocessing', 'cond_extraction', 'sparse_sampling', 'shape_slat_sampling',
                'tex_slat_sampling', 'latent_decoding', 'mesh_simplify', 'pbr_render',
                'glb_fill_holes', 'glb_bvh', 'glb_remesh', 'glb_uv_unwrap', 'glb_texture_bake', 'glb_finalize', 'glb_export']:
        if key in warmup:
            print(f"{key:<35} {warmup[key]:<12.1f}")
    print("-" * 80)
    print(f"{'TOTAL':<35} {warmup['total']:<12.1f}")
    print(f"{'Peak Memory (GB)':<35} {warmup.get('peak_memory_gb', 0):<12.2f}")
    
    # Average
    print()
    print("AVERAGE (Excluding Warmup)")
    print("-" * 80)
    avg = data['average']
    print(f"{'Step':<35} {'Time (ms)':<12}")
    print("-" * 80)
    for key in ['preprocessing', 'cond_extraction', 'sparse_sampling', 'shape_slat_sampling',
                'tex_slat_sampling', 'latent_decoding', 'mesh_simplify', 'pbr_render',
                'glb_fill_holes', 'glb_bvh', 'glb_remesh', 'glb_uv_unwrap', 'glb_texture_bake', 'glb_finalize', 'glb_export']:
        if key in avg:
            print(f"{key:<35} {avg[key]:<12.1f}")
    print("-" * 80)
    print(f"{'TOTAL':<35} {avg['total']:<12.1f}")
    print(f"{'Peak Memory (GB)':<35} {avg.get('peak_memory_gb', 0):<12.2f}")

if __name__ == "__main__":
    data = run_benchmark()
    print_results(data)
    
    # Save to JSON
    output_path = OUTPUT_DIR / "benchmark_fork.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")