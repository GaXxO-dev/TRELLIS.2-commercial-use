#!/usr/bin/env python3
"""
Verify the cubemap conversion is correct.
"""
import torch
import numpy as np

# Test the actual cube_to_dir implementations
def cube_to_dir_old(s, x, y):
    """Old fork version - Y/Z swapped in some faces."""
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def cube_to_dir_official(s, x, y):
    """Official nvdiffrast version."""
    if s == 0:   rx, ry, rz = torch.ones_like(x), -x, -y
    elif s == 1: rx, ry, rz = -torch.ones_like(x), x, -y
    elif s == 2: rx, ry, rz = x, y, torch.ones_like(x)
    elif s == 3: rx, ry, rz = x, -y, -torch.ones_like(x)
    elif s == 4: rx, ry, rz = x, torch.ones_like(x), -y
    elif s == 5: rx, ry, rz = -x, -torch.ones_like(x), -y
    return torch.stack((rx, ry, rz), dim=-1)

print("Testing with x=0.25, y=0.75 (different values):")
print("=" * 60)
x = torch.tensor([0.25])
y = torch.tensor([0.75])

for s in range(6):
    old_dir = cube_to_dir_old(s, x, y)
    new_dir = cube_to_dir_official(s, x, y)
    face_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    print(f"{face_names[s]}: old={old_dir[0].tolist()}, new={new_dir[0].tolist()}")
    if not torch.allclose(old_dir, new_dir):
        print(f"       DIFFERENT!")

print("\n" + "=" * 60)
print("Convention comparison:")
print("\nOfficial cube_to_dir:")
print("  +X: (1, -x, -y) -> at center (0,0): (1, 0, 0)")
print("  -X: (-1, x, -y) -> at center: (-1, 0, 0)")
print("  +Y: (x, y, 1) -> at center: (0, 0, 1)")
print("  -Y: (x, -y, -1) -> at center: (0, 0, -1)")
print("  +Z: (x, 1, -y) -> at center: (0, 1, 0)")
print("  -Z: (-x, -1, -y) -> at center: (0, -1, 0)")
print("\nThe issue: Fork had Y and Y swapped in many faces.")
print("This means the environment light is sampled incorrectly,")
print("causing lighting direction mismatch.")