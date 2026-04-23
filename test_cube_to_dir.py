#!/usr/bin/env python3
"""
Test if cube_to_dir is working correctly for environment map conversion.
"""
import torch
import numpy as np

def cube_to_dir_old(s, x, y):
    """Old fork version."""
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def cube_to_dir_new(s, x, y):
    """New official version."""
    if s == 0:   rx, ry, rz = torch.ones_like(x), -x, -y
    elif s == 1: rx, ry, rz = -torch.ones_like(x), x, -y
    elif s == 2: rx, ry, rz = x, y, torch.ones_like(x)
    elif s == 3: rx, ry, rz = x, -y, -torch.ones_like(x)
    elif s == 4: rx, ry, rz = x, torch.ones_like(x), -y
    elif s == 5: rx, ry, rz = -x, -torch.ones_like(x), -y
    return torch.stack((rx, ry, rz), dim=-1)

print("Testing cube_to_dir differences:")
print("=" * 60)

x = torch.tensor([0.5])
y = torch.tensor([0.5])

for s in range(6):
    old_dir = cube_to_dir_old(s, x, y)
    new_dir = cube_to_dir_new(s, x, y)
    diff = (old_dir - new_dir).abs().max().item()
    print(f"Face {s}: old={old_dir[0].tolist()}, new={new_dir[0].tolist()}, diff={diff:.4f}")

print("\n" + "=" * 60)
print("Face conventions:")
print("  0: +X")
print("  1: -X")
print("  2: +Y")
print("  3: -Y")
print("  4: +Z")
print("  5: -Z")
print()
print("OLD: Y and Y, swapped in some faces")
print("NEW: Official nvdiffrast/nvdiffrec convention")
print()
print("If the cubemap is built differently, lighting directions will")
print("differ. This affects specular reflection and diffuse irradiance.")
print()
print("The issue is that latlong_to_cubemap needs to use the SAME")
print("convention as when sampling the cubemap during shade().")