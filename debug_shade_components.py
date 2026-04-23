#!/usr/bin/env python3
"""
Debug script to compare shade() intermediate values between fork and official.

This script captures:
1. Diffuse irradiance lookup
2. Specular prefilter values  
3. FG LUT values
4. NdotV values
5. Reflection vectors
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
import numpy as np
from PIL import Image

# Check for saved debug data
FORK_DIR = "test_output_pbr"
OFFICIAL_DIR = "test_output_pbr_official"

def load_debug_tensor(path):
    """Load a debug tensor if it exists."""
    if os.path.exists(path):
        return np.load(path)
    return None

def compare_tensors(name, fork_data, official_data, mask=None):
    """Compare two tensors and print statistics."""
    if fork_data is None or official_data is None:
        print(f"  {name}: MISSING DATA")
        return
    
    if mask is not None:
        fork_data = fork_data[mask]
        official_data = official_data[mask]
    
    fork_flat = fork_data.flatten()
    official_flat = official_data.flatten()
    
    fork_mean = fork_flat.mean()
    official_mean = official_flat.mean()
    
    if official_mean > 1e-6:
        ratio = fork_mean / official_mean
    else:
        ratio = float('inf') if fork_mean > 1e-6 else 1.0
    
    print(f"  {name}:")
    print(f"    Fork mean:     {fork_mean:.6f} std: {fork_flat.std():.6f}")
    print(f"    Official mean: {official_mean:.6f} std: {official_flat.std():.6f}")
    print(f"    Ratio:         {ratio:.4f}")

def main():
    print("=" * 70)
    print("SHADE() COMPONENT COMPARISON")
    print("=" * 70)
    
    # Load mask for geometry filtering
    fork_mask = np.array(Image.open(f"{FORK_DIR}/mask_pbr_00.png")).astype(np.float32) / 255.0
    official_mask = np.array(Image.open(f"{OFFICIAL_DIR}/mask_pbr_00.png")).astype(np.float32) / 255.0
    
    if fork_mask.ndim == 3:
        fork_mask = fork_mask.max(axis=-1) > 0
    else:
        fork_mask = fork_mask > 0
    
    if official_mask.ndim == 3:
        official_mask = official_mask.max(axis=-1) > 0
    else:
        official_mask = official_mask > 0
    
    # Load existing data
    fork_shaded = np.array(Image.open(f"{FORK_DIR}/shaded_00.png")).astype(np.float32) / 255.0
    official_shaded = np.array(Image.open(f"{OFFICIAL_DIR}/shaded_00.png")).astype(np.float32) / 255.0
    
    fork_clay = np.array(Image.open(f"{FORK_DIR}/clay_00.png")).astype(np.float32) / 255.0
    official_clay = np.array(Image.open(f"{OFFICIAL_DIR}/clay_00.png")).astype(np.float32) / 255.0
    
    fork_base = np.array(Image.open(f"{FORK_DIR}/base_color_00.png")).astype(np.float32) / 255.0
    official_base = np.array(Image.open(f"{OFFICIAL_DIR}/base_color_00.png")).astype(np.float32) / 255.0
    
    fork_metallic = np.array(Image.open(f"{FORK_DIR}/metallic_00.png")).astype(np.float32) / 255.0
    official_metallic = np.array(Image.open(f"{OFFICIAL_DIR}/metallic_00.png")).astype(np.float32) / 255.0
    
    fork_roughness = np.array(Image.open(f"{FORK_DIR}/roughness_00.png")).astype(np.float32) / 255.0
    official_roughness = np.array(Image.open(f"{OFFICIAL_DIR}/roughness_00.png")).astype(np.float32) / 255.0
    
    fork_normal = np.array(Image.open(f"{FORK_DIR}/normal_pbr_00.png")).astype(np.float32) / 255.0
    official_normal = np.array(Image.open(f"{OFFICIAL_DIR}/normal_pbr_00.png")).astype(np.float32) / 255.0
    
    # Convert normal from [0,1] to [-1,1]
    fork_normal_3d = fork_normal * 2 - 1
    official_normal_3d = official_normal * 2 - 1
    
    # Compute luminance
    def luminance(img):
        if img.ndim == 3:
            return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        return img
    
    fork_lum = luminance(fork_shaded)
    official_lum = luminance(official_shaded)
    
    fork_clay_2d = fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay
    official_clay_2d = official_clay[..., 0] if official_clay.ndim == 3 else official_clay
    
    # Pre-SSAO computation
    fork_valid = fork_mask & (fork_clay_2d > 0.01)
    official_valid = official_mask & (official_clay_2d > 0.01)
    
    fork_pre = fork_lum[fork_valid] / fork_clay_2d[fork_valid]
    official_pre = official_lum[official_valid] / official_clay_2d[official_valid]
    
    print("\n1. FINAL OUTPUT COMPARISON:")
    compare_tensors("Shaded luminance", fork_lum, official_lum, fork_mask)
    
    print("\n2. PRE-SSAO COMPARISON:")
    compare_tensors("Pre-SSAO", fork_pre, official_pre, None)
    
    print("\n3. MATERIAL PROPERTIES:")
    compare_tensors("Base color", fork_base, official_base, fork_mask)
    compare_tensors("Metallic", fork_metallic, official_metallic, fork_mask)
    compare_tensors("Roughness", fork_roughness, official_roughness, fork_mask)
    
    print("\n4. NORMAL COMPARISON:")
    # Compare normal statistics rather than direct difference due to different mask sizes
    fork_n = fork_normal_3d[fork_mask]
    official_n = official_normal_3d[official_mask]
    
    print(f"  Fork normal mean:     {fork_n.mean(axis=0)}")
    print(f"  Official normal mean: {official_n.mean(axis=0)}")
    
    # For pixels that overlap in both masks
    overlap_mask = fork_mask & official_mask
    if overlap_mask.sum() > 0:
        fork_n_overlap = fork_normal_3d[overlap_mask]
        official_n_overlap = official_normal_3d[overlap_mask]
        dot_product = np.sum(fork_n_overlap * official_n_overlap, axis=-1)
        dot_product = np.clip(dot_product, -1, 1)
        angle_diff = np.arccos(dot_product) * 180 / np.pi
        
        print(f"  Overlapping pixels: {overlap_mask.sum()}")
        print(f"  Normal angle difference:")
        print(f"    Mean:   {angle_diff.mean():.4f}°")
        print(f"    Std:    {angle_diff.std():.4f}°")
        print(f"    Median: {np.median(angle_diff):.4f}°")
        print(f"    Within 10°: {100 * (angle_diff < 10).mean():.1f}%")
    
    print("\n5. SSAO ANALYSIS:")
    fork_f_occ = 1 - fork_clay_2d
    official_f_occ = 1 - official_clay_2d
    compare_tensors("f_occ", fork_f_occ, official_f_occ, fork_mask)
    
    print("\n" + "=" * 70)
    print("KEY FINDING: Compare pre-SSAO brightness ratio")
    print("=" * 70)
    
    pre_ratio = fork_pre.mean() / official_pre.mean()
    print(f"Pre-SSAO brightness ratio: {pre_ratio:.4f}")
    print(f"This ~{100*(1-pre_ratio):.1f}% difference is in the PBR shading.")
    print()
    print("POTENTIAL CAUSES:")
    print("  1. Missing spec = lerp(spec, diffuse, roughness) in shade()")
    print("  2. Different cubemap sampling (diffuse irradiance)")
    print("  3. Different specular prefilter values")
    print("  4. Different FG LUT sampling")
    print("  5. Normal-based lighting direction differences")

if __name__ == "__main__":
    main()