#!/usr/bin/env python3
"""
Investigate why normals differ so much between fork and official.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image

FORK_DIR = "test_output_pbr"
OFFICIAL_DIR = "test_output_pbr_official"

def load_img(path):
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path))
    return arr.astype(np.float32) / 255.0

print("=" * 70)
print("NORMAL ANGLE INVESTIGATION")
print("=" * 70)

# Load normal maps (camera space, encoded as [0,255] for [-1,1])
fork_normal = load_img(f"{FORK_DIR}/normal_pbr_00.png")
official_normal = load_img(f"{OFFICIAL_DIR}/normal_pbr_00.png")
fork_mask = load_img(f"{FORK_DIR}/mask_pbr_00.png")
official_mask = load_img(f"{OFFICIAL_DIR}/mask_pbr_00.png")

# Get geometry mask
fork_geom = fork_mask.max(axis=-1) > 0 if fork_mask.ndim == 3 else fork_mask > 0
official_geom = official_mask.max(axis=-1) > 0 if official_mask.ndim == 3 else official_mask > 0

print(f"\nGeometry mask:")
print(f"  Fork geometry pixels: {fork_geom.sum()}")
print(f"  Official geometry pixels: {official_geom.sum()}")

# Decode normals
fork_normal_decoded = fork_normal * 2 - 1
official_normal_decoded = official_normal * 2 - 1

# Compute per-pixel angle difference
def compute_angle_diff(n1, n2):
    dot = np.sum(n1 * n2, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot) * 180 / np.pi

# Only compare where both have geometry
both_geom = fork_geom & official_geom
angle_diff = compute_angle_diff(fork_normal_decoded, official_normal_decoded)

print(f"\nNormal angle difference (where both have geometry):")
print(f"  Mean: {angle_diff[both_geom].mean():.2f}°")
print(f"  Median: {np.median(angle_diff[both_geom]):.2f}°")
print(f"  Std: {angle_diff[both_geom].std():.2f}°")
print(f"  Max: {angle_diff[both_geom].max():.2f}°")
print(f"  90th percentile: {np.percentile(angle_diff[both_geom], 90):.2f}°")
print(f"  95th percentile: {np.percentile(angle_diff[both_geom], 95):.2f}°")

# Histogram
print(f"\nNormal angle difference distribution:")
for thresh in [0, 5, 10, 15, 20, 30, 45, 60, 90]:
    count = (angle_diff[both_geom] <= thresh).sum()
    pct = 100 * count / both_geom.sum()
    print(f"  <= {thresh:3d}°: {pct:5.1f}%")

# Analyze specific regions - center vs edges
H, W = angle_diff.shape
cy, cx = H // 2, W // 2
center_mask = np.zeros_like(both_geom, dtype=bool)
center_mask[max(0, cy-50):min(H, cy+50), max(0, cx-50):min(W, cx+50)] = True
center_geom = both_geom & center_mask

print(f"\nCenter region normal angle difference (center 100x100):")
if center_geom.sum() > 0:
    print(f"  Mean: {angle_diff[center_geom].mean():.2f}°")
    print(f"  Median: {np.median(angle_diff[center_geom]):.2f}°")
else:
    print("  No center geometry pixels")

# Compare normal values directly
print(f"\nNormal value comparison (geometry pixels only):")
fork_n = fork_normal_decoded[both_geom]
official_n = official_normal_decoded[both_geom]
print(f"  Fork X:     mean={fork_n[:, 0].mean():.4f}, std={fork_n[:, 0].std():.4f}")
print(f"  Official X: mean={official_n[:, 0].mean():.4f}, std={official_n[:, 0].std():.4f}")
print(f"  Fork Y:     mean={fork_n[:, 1].mean():.4f}, std={fork_n[:, 1].std():.4f}")
print(f"  Official Y: mean={official_n[:, 1].mean():.4f}, std={official_n[:, 1].std():.4f}")
print(f"  Fork Z:     mean={fork_n[:, 2].mean():.4f}, std={fork_n[:, 2].std():.4f}")
print(f"  Official Z: mean={official_n[:, 2].mean():.4f}, std={official_n[:, 2].std():.4f}")

# Compute signed difference
diff_x = fork_n[:, 0] - official_n[:, 0]
diff_y = fork_n[:, 1] - official_n[:, 1]
diff_z = fork_n[:, 2] - official_n[:, 2]

print(f"\nNormal difference (fork - official):")
print(f"  X diff: mean={diff_x.mean():.4f}, std={diff_x.std():.4f}")
print(f"  Y diff: mean={diff_y.mean():.4f}, std={diff_y.std():.4f}")
print(f"  Z diff: mean={diff_z.mean():.4f}, std={diff_z.std():.4f}")

# Check if normals need to be re-normalized after interpolation
fork_mag = np.linalg.norm(fork_n, axis=-1)
official_mag = np.linalg.norm(official_n, axis=-1)
print(f"\nNormal magnitude:")
print(f"  Fork:     mean={fork_mag.mean():.6f}, std={fork_mag.std():.6f}")
print(f"  Official: mean={official_mag.mean():.6f}, std={official_mag.std():.6f}")
print(f"  (Both should be ~1.0 if properly normalized)")