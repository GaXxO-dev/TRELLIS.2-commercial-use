#!/usr/bin/env python
"""
Compare inference speed between Fork (DRTK) and Official (nvdiffrast).
Loads benchmark results from JSON files and produces comparison report.
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Implementation status for each step
IMPL_STATUS = {
    'preprocessing': 'SAME',
    'cond_extraction': 'SAME',
    'sparse_sampling': 'SAME',
    'shape_slat_sampling': 'SAME',
    'tex_slat_sampling': 'SAME',
    'latent_decoding': 'SAME',
    'mesh_simplify': 'SAME',
    'pbr_render': 'DIFFERENT',
    'glb_fill_holes': 'SAME',
    'glb_bvh': 'SAME',
    'glb_remesh': 'SAME',
    'glb_uv_unwrap': 'SAME',
    'glb_texture_bake': 'DIFFERENT',
    'glb_finalize': 'SAME',
    'glb_export': 'MIXED',
}

STEP_NAMES = {
    'preprocessing': 'Image Preprocessing',
    'cond_extraction': 'Conditioning Extraction',
    'sparse_sampling': 'Sparse Structure Sampling',
    'shape_slat_sampling': 'Shape SLat Sampling',
    'tex_slat_sampling': 'Texture SLat Sampling',
    'latent_decoding': 'Latent Decoding',
    'mesh_simplify': 'Mesh Simplification',
    'pbr_render': 'PBR Preview Rendering',
    'glb_fill_holes': 'GLB - Fill Holes',
    'glb_bvh': 'GLB - Build BVH',
    'glb_remesh': 'GLB - Remesh',
    'glb_uv_unwrap': 'GLB - UV Unwrap',
    'glb_texture_bake': 'GLB - Texture Bake',
    'glb_finalize': 'GLB - Finalize',
    'glb_export': 'GLB Export (total)',
}

def load_results():
    fork_path = SCRIPT_DIR / "benchmark_fork.json"
    official_path = SCRIPT_DIR / "benchmark_official.json"
    
    if not fork_path.exists():
        print(f"ERROR: Fork benchmark results not found: {fork_path}")
        return None, None
    
    if not official_path.exists():
        print(f"ERROR: Official benchmark results not found: {official_path}")
        return None, None
    
    with open(fork_path) as f:
        fork_data = json.load(f)
    
    with open(official_path) as f:
        official_data = json.load(f)
    
    return fork_data, official_data

def compute_diff(fork_val, official_val):
    if official_val == 0:
        return 0.0
    return ((fork_val - official_val) / official_val) * 100

def print_comparison(fork_data, official_data):
    print("=" * 100)
    print("INFERENCE SPEED BENCHMARK: Fork (DRTK) vs Official (nvdiffrast)")
    print("=" * 100)
    
    config = fork_data['config']
    print(f"\nResolution: {config['resolution']} | PBR Preview: {config['pbr_views']} views @ {config['pbr_resolution']}px | Texture: {config['texture_size']}px")
    print(f"Iterations: {config['iterations']} (1 warmup + {config['iterations']-1} measured)")
    
    # Warmup comparison
    print("\n" + "=" * 100)
    print("WARMUP (First Iteration)")
    print("=" * 100)
    print(f"{'Step':<40} {'Fork (ms)':<14} {'Official (ms)':<14} {'Diff (%)':<12} {'Impl':<12}")
    print("-" * 100)
    
    fork_warmup = fork_data['warmup']
    official_warmup = official_data['warmup']
    
    ordered_keys = ['preprocessing', 'cond_extraction', 'sparse_sampling', 'shape_slat_sampling',
                    'tex_slat_sampling', 'latent_decoding', 'mesh_simplify', 'pbr_render',
                    'glb_fill_holes', 'glb_bvh', 'glb_remesh', 'glb_uv_unwrap', 'glb_texture_bake', 'glb_finalize', 'glb_export']
    
    for key in ordered_keys:
        fork_val = fork_warmup.get(key, 0)
        official_val = official_warmup.get(key, 0)
        diff = compute_diff(fork_val, official_val)
        impl = IMPL_STATUS.get(key, 'SAME')
        name = STEP_NAMES.get(key, key)
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        print(f"{name:<40} {fork_val:<14.1f} {official_val:<14.1f} {diff_str:<12} {impl:<12}")
    
    print("-" * 100)
    fork_total = fork_warmup['total']
    official_total = official_warmup['total']
    diff_total = compute_diff(fork_total, official_total)
    diff_str = f"+{diff_total:.1f}%" if diff_total >= 0 else f"{diff_total:.1f}%"
    print(f"{'TOTAL':<40} {fork_total:<14.1f} {official_total:<14.1f} {diff_str:<12}")
    
    fork_mem = fork_warmup.get('peak_memory_gb', 0)
    official_mem = official_warmup.get('peak_memory_gb', 0)
    mem_diff = compute_diff(fork_mem, official_mem)
    mem_diff_str = f"+{mem_diff:.1f}%" if mem_diff >= 0 else f"{mem_diff:.1f}%"
    print(f"{'Peak Memory (GB)':<40} {fork_mem:<14.2f} {official_mem:<14.2f} {mem_diff_str:<12}")
    
    # Average comparison
    print("\n" + "=" * 100)
    print("AVERAGE (Excluding Warmup)")
    print("=" * 100)
    print(f"{'Step':<40} {'Fork (ms)':<14} {'Official (ms)':<14} {'Diff (%)':<12} {'Impl':<12}")
    print("-" * 100)
    
    fork_avg = fork_data['average']
    official_avg = official_data['average']
    
    for key in ordered_keys:
        fork_val = fork_avg.get(key, 0)
        official_val = official_avg.get(key, 0)
        diff = compute_diff(fork_val, official_val)
        impl = IMPL_STATUS.get(key, 'SAME')
        name = STEP_NAMES.get(key, key)
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        print(f"{name:<40} {fork_val:<14.1f} {official_val:<14.1f} {diff_str:<12} {impl:<12}")
    
    print("-" * 100)
    fork_total = fork_avg['total']
    official_total = official_avg['total']
    diff_total = compute_diff(fork_total, official_total)
    diff_str = f"+{diff_total:.1f}%" if diff_total >= 0 else f"{diff_total:.1f}%"
    print(f"{'TOTAL':<40} {fork_total:<14.1f} {official_total:<14.1f} {diff_str:<12}")
    
    fork_mem = fork_avg.get('peak_memory_gb', 0)
    official_mem = official_avg.get('peak_memory_gb', 0)
    mem_diff = compute_diff(fork_mem, official_mem)
    mem_diff_str = f"+{mem_diff:.1f}%" if mem_diff >= 0 else f"{mem_diff:.1f}%"
    print(f"{'Peak Memory (GB)':<40} {fork_mem:<14.2f} {official_mem:<14.2f} {mem_diff_str:<12}")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    faster_slower = "SLOWER" if diff_total > 0 else "FASTER"
    print(f"\nTotal inference time:")
    print(f"  Fork:     {fork_total:.1f} ms ({fork_total/1000:.2f} seconds)")
    print(f"  Official: {official_total:.1f} ms ({official_total/1000:.2f} seconds)")
    print(f"  Fork is {abs(diff_total):.1f}% {faster_slower}")
    
    # PBR Rendering comparison
    pbr_fork = fork_avg.get('pbr_render', 0)
    pbr_official = official_avg.get('pbr_render', 0)
    pbr_diff = compute_diff(pbr_fork, pbr_official)
    pbr_faster_slower = "SLOWER" if pbr_diff > 0 else "FASTER"
    print(f"\nPBR Preview Rendering (Pure PyTorch vs nvdiffrec):")
    print(f"  Fork (Pure PyTorch): {pbr_fork:.1f} ms")
    print(f"  Official (nvdiffrec): {pbr_official:.1f} ms")
    print(f"  Fork is {abs(pbr_diff):.1f}% {pbr_faster_slower}")
    
    # GLB Texture Bake comparison
    tex_fork = fork_avg.get('glb_texture_bake', 0)
    tex_official = official_avg.get('glb_texture_bake', 0)
    tex_diff = compute_diff(tex_fork, tex_official)
    tex_faster_slower = "SLOWER" if tex_diff > 0 else "FASTER"
    print(f"\nGLB Texture Bake (DRTK vs nvdiffrast):")
    print(f"  Fork (DRTK):        {tex_fork:.1f} ms")
    print(f"  Official (nvdiffrast): {tex_official:.1f} ms")
    print(f"  Fork is {abs(tex_diff):.1f}% {tex_faster_slower}")
    
    print(f"\nCUDA Memory:")
    print(f"  Fork:     {fork_mem:.2f} GB peak")
    print(f"  Official: {official_mem:.2f} GB peak")
    print(f"  Diff:     {mem_diff_str}")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    fork_data, official_data = load_results()
    if fork_data and official_data:
        print_comparison(fork_data, official_data)