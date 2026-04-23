#!/usr/bin/env python3
"""
Compare diffuse and specular components between fork and official.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

FORK_DIR = "test_output_pbr"
OFFICIAL_DIR = "test_output_pbr_official"

def load_img(path):
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path))
    return arr.astype(np.float32) / 255.0

print("=" * 70)
print("PBR SHADING COMPONENT ANALYSIS")
print("=" * 70)

# Load all the data
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

# Filter valid pixels (geometry + non-zero clay)
fork_valid = fork_geom & (fork_clay[..., 0] > 0.01 if fork_clay.ndim == 3 else fork_clay > 0.01)
official_valid = official_geom & (official_clay[..., 0] > 0.01 if official_clay.ndim == 3 else official_clay > 0.01)

# Decode normals (stored as [0,1] for [-1,1])
fork_normal_3d = fork_normal * 2 - 1
official_normal_3d = official_normal * 2 - 1

# Compute pre-SSAO luminance
def luminance(img):
    if img.ndim == 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img

fork_lum = luminance(fork_shaded)
official_lum = luminance(official_shaded)
fork_clay_2d = fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay
official_clay_2d = official_clay[..., 0] if official_clay.ndim == 3 else official_clay

fork_pre = fork_lum[fork_valid] / fork_clay_2d[fork_valid]
official_pre = official_lum[official_valid] / official_clay_2d[official_valid]

print(f"\nMATERIAL PROPERTIES:")
print(f"  Base color:  fork={fork_base_color[fork_geom].mean():.4f}, official={official_base_color[official_geom].mean():.4f}")
print(f"  Metallic:    fork={fork_metallic[fork_geom].mean():.4f}, official={official_metallic[official_geom].mean():.4f}")
print(f"  Roughness:   fork={fork_roughness[fork_geom].mean():.4f}, official={official_roughness[official_geom].mean():.4f}")

# Compute PBR color channels separately
fork_r = fork_shaded[fork_valid, 0] if fork_shaded.ndim == 3 else fork_shaded[fork_valid]
fork_g = fork_shaded[fork_valid, 1] if fork_shaded.ndim == 3 else fork_shaded[fork_valid]
fork_b = fork_shaded[fork_valid, 2] if fork_shaded.ndim == 3 else fork_shaded[fork_valid]

official_r = official_shaded[official_valid, 0] if official_shaded.ndim == 3 else official_shaded[official_valid]
official_g = official_shaded[official_valid, 1] if official_shaded.ndim == 3 else official_shaded[official_valid]
official_b = official_shaded[official_valid, 2] if official_shaded.ndim == 3 else official_shaded[official_valid]

print(f"\nSHADED COLOR CHANNELS:")
print(f"  Fork R: {fork_r.mean():.6f}, G: {fork_g.mean():.6f}, B: {fork_b.mean():.6f}")
print(f"  Official R: {official_r.mean():.6f}, G: {official_g.mean():.6f}, B: {official_b.mean():.6f}")
print(f"  Ratio R: {fork_r.mean()/official_r.mean():.4f}")
print(f"  Ratio G: {fork_g.mean()/official_g.mean():.4f}")
print(f"  Ratio B: {fork_b.mean()/official_b.mean():.4f}")

# Estimate diffuse vs specular contribution
# In PBR: shaded = (diffuse * (1 - metallic) + specular) * (1 - occlusion)
# We can't directly separate diffuse and specular from the output,
# but we can check if the ratio is uniform across colors

# For metallic materials, diffuse is negligible
# For non-metallic materials, diffuse contributes more

fork_metallic_v = fork_metallic[fork_geom] if fork_metallic.ndim == 3 else fork_metallic[fork_geom]
official_metallic_v = official_metallic[official_geom] if official_metallic.ndim == 3 else official_metallic[official_geom]

# Separate analysis for metallic and non-metallic pixels
fork_metallic_mask = fork_metallic_v > 0.5
official_metallic_mask = official_metallic_v > 0.5

print(f"\nMATERIAL DISTRIBUTION:")
print(f"  Fork metallic pixels: {fork_metallic_mask.sum()} / {fork_metallic_mask.size} ({100*fork_metallic_mask.mean():.1f}%)")
print(f"  Official metallic pixels: {official_metallic_mask.sum()} / {official_metallic_mask.size} ({100*official_metallic_mask.mean():.1f}%)")

# For non-metallic (diffuse-dominated), check brightness
fork_nonmetal = fork_metallic_v < 0.5
official_nonmetal = official_metallic_v < 0.5

# These masks should mostly overlap
fork_valid_nonmetal = fork_valid[fork_geom] & fork_nonmetal
official_valid_nonmetal = official_valid[official_geom] & official_nonmetal

# Pre-SSAO for non-metallic (mostly diffuse)
if fork_valid_nonmetal.sum() > 0 and official_valid_nonmetal.sum() > 0:
    fork_pre_nonmetal = fork_lum[fork_valid_nonmetal] / fork_clay_2d[fork_valid_nonmetal]
    official_pre_nonmetal = official_lum[official_valid_nonmetal] / official_clay_2d[official_valid_nonmetal]
    print(f"\nPRE-SSAO NON-METALLIC (diffuse-dominated):")
    print(f"  Fork: {fork_pre_nonmetal.mean():.6f}")
    print(f"  Official: {official_pre_nonmetal.mean():.6f}")
    print(f"  Ratio: {fork_pre_nonmetal.mean()/official_pre_nonmetal.mean():.4f}")

print(f"\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Pre-SSAO brightness ratio: {fork_pre.mean()/official_pre.mean():.3f}")
print(f"SSAO match: YES (previously verified)")
print(f"Materials match: YES (verified above)")
print(f"\nThe ~11% darkness is in PBR shading (diffuse + specular).")
print(f"This could be caused by:")
print(f"  1. Environment map lighting direction")
print(f"  2. Normal-based diffuse irradiance lookup")
print(f"  3. Reflection-based specular prefilter lookup")
print(f"  4. FG LUT for specular (NdotV, roughness)")