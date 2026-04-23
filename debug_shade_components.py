#!/usr/bin/env python3
"""
Debug shade components by loading and comparing saved debug data.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image
import sys

FORK_DIR = "test_output_pbr"
OFFICIAL_DIR = "test_output_pbr_official"

def load_img(path):
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path))
    return arr.astype(np.float32) / 255.0

def load_npy(path):
    if not os.path.exists(path):
        return None
    return np.load(path)

print("=" * 70)
print("SHADE COMPONENT DEBUG")
print("=" * 70)

# Load images
fork_shaded = load_img(f"{FORK_DIR}/shaded_00.png")
fork_clay = load_img(f"{FORK_DIR}/clay_00.png")
fork_base_color = load_img(f"{FORK_DIR}/base_color_00.png")
fork_metallic = load_img(f"{FORK_DIR}/metallic_00.png")
fork_roughness = load_img(f"{FORK_DIR}/roughness_00.png")
fork_normal = load_img(f"{FORK_DIR}/normal_pbr_00.png")
fork_mask = load_img(f"{FORK_DIR}/mask_pbr_00.png")

official_shaded = load_img(f"{OFFICIAL_DIR}/shaded_00.png")
official_clay = load_img(f"{OFFICIAL_DIR}/clay_00.png")
official_base_color = load_img(f"{OFFICIAL_DIR}/base_color_00.png")
official_metallic = load_img(f"{OFFICIAL_DIR}/metallic_00.png")
official_roughness = load_img(f"{OFFICIAL_DIR}/roughness_00.png")
official_normal = load_img(f"{OFFICIAL_DIR}/normal_pbr_00.png")
official_mask = load_img(f"{OFFICIAL_DIR}/mask_pbr_00.png")

# Get geometry mask
fork_geom = fork_mask.max(axis=-1) > 0 if fork_mask.ndim == 3 else fork_mask > 0
official_geom = official_mask.max(axis=-1) > 0 if official_mask.ndim == 3 else official_mask > 0

# Decode normals from [0,1] to [-1,1]
fork_normal_decoded = fork_normal * 2 - 1
official_normal_decoded = official_normal * 2 - 1

# Compute normal magnitude (should be ~1.0 if normalized)
fork_normal_mag = np.linalg.norm(fork_normal_decoded, axis=-1)
official_normal_mag = np.linalg.norm(official_normal_decoded, axis=-1)

print("\nNORMAL ANALYSIS:")
print(f"  Fork normal magnitude:     mean={fork_normal_mag[fork_geom].mean():.6f}, std={fork_normal_mag[fork_geom].std():.6f}")
print(f"  Official normal magnitude: mean={official_normal_mag[official_geom].mean():.6f}, std={official_normal_mag[official_geom].std():.6f}")

# Compute normal angle difference
fork_normal_geom = fork_normal_decoded[fork_geom]
official_normal_geom = official_normal_decoded[official_geom]

# The normals at geometry pixels should be very similar
# Compute dot product to find angular difference
min_len = min(len(fork_normal_geom), len(official_normal_geom))
fork_normal_trim = fork_normal_geom[:min_len]
official_normal_trim = official_normal_geom[:min_len]

dot_product = np.sum(fork_normal_trim * official_normal_trim, axis=-1)
angle_diff = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi

print(f"  Normal angle difference:   mean={angle_diff.mean():.2f}°, median={np.median(angle_diff):.2f}°, max={angle_diff.max():.2f}°")

# Per-channel normal difference
normal_diff = np.abs(fork_normal_decoded - official_normal_decoded)
print(f"  Normal per-channel diff:   R={normal_diff[fork_geom, 0].mean():.4f}, G={normal_diff[fork_geom, 1].mean():.4f}, B={normal_diff[fork_geom, 2].mean():.4f}")

# Material channels
print("\nMATERIAL CHANNELS:")
print(f"  Base color:  fork={fork_base_color[fork_geom].mean():.4f}, official={official_base_color[official_geom].mean():.4f}, ratio={fork_base_color[fork_geom].mean()/official_base_color[official_geom].mean():.4f}")
print(f"  Metallic:    fork={fork_metallic[fork_geom].mean():.4f}, official={official_metallic[official_geom].mean():.4f}, ratio={fork_metallic[fork_geom].mean()/official_metallic[official_geom].mean():.4f}")
print(f"  Roughness:   fork={fork_roughness[fork_geom].mean():.4f}, official={official_roughness[official_geom].mean():.4f}, ratio={fork_roughness[fork_geom].mean()/official_roughness[official_geom].mean():.4f}")

# Compute luminance
def lum(img):
    if img.ndim == 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img

fork_lum = lum(fork_shaded)
official_lum = lum(official_shaded)

fork_clay_2d = fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay
official_clay_2d = official_clay[..., 0] if official_clay.ndim == 3 else official_clay

# Filter to valid geometry with non-zero clay
fork_valid = fork_geom & (fork_clay_2d > 0.01)
official_valid = official_geom & (official_clay_2d > 0.01)

# Estimate pre-SSAO lighting
fork_pre = fork_lum[fork_valid] / fork_clay_2d[fork_valid]
official_pre = official_lum[official_valid] / official_clay_2d[official_valid]

print("\nPRE-SSAO SHADING (estimated):")
print(f"  Fork:     mean={fork_pre.mean():.6f}")
print(f"  Official: mean={official_pre.mean():.6f}")
print(f"  Ratio: {fork_pre.mean() / official_pre.mean():.4f}")
print(f"  (Fork is {(1 - fork_pre.mean() / official_pre.mean()) * 100:.1f}% darker before SSAO)")

# Analyze color channels separately
print("\nPER-CHANNEL SHADED OUTPUT:")
for c, name in enumerate(['R', 'G', 'B']):
    fork_chan = fork_shaded[fork_valid[..., None] if fork_shaded.ndim == 3 else fork_valid, c if fork_shaded.ndim == 3 else 0]
    official_chan = official_shaded[official_valid[..., None] if official_shaded.ndim == 3 else official_valid, c if official_shaded.ndim == 3 else 0]
    ratio = fork_chan.mean() / official_chan.mean() if official_chan.mean() > 0 else 0
    print(f"  {name}: fork={fork_chan.mean():.6f}, official={official_chan.mean():.6f}, ratio={ratio:.4f}")

# Check if darker uniformly across channels or if one channel is more affected
print("\nCHANNEL-WISE PRE-SSAO:")
for c, name in enumerate(['R', 'G', 'B']):
    fork_pre_chan = fork_shaded[..., c][fork_valid] / fork_clay_2d[fork_valid]
    official_pre_chan = official_shaded[..., c][official_valid] / official_clay_2d[official_valid]
    ratio = fork_pre_chan.mean() / official_pre_chan.mean()
    print(f"  {name}: fork={fork_pre_chan.mean():.6f}, official={official_pre_chan.mean():.6f}, ratio={ratio:.4f}")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("=" * 70)
print("The ~11% darkness is in the pre-SSAO shading.")
print("This could be caused by:")
print("  1. Diffuse irradiance lookup - different normal direction")
print("  2. Specular prefilter Lookup - different reflection direction")
print("  3. FG LUT sampling - different NdotV due to normal difference")
print("  4. Normal interpolation - DRTK vs nvdiffrast edge handling")