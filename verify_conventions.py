#!/usr/bin/env python3
"""Verify cube_to_dir conventions between fork and official."""
import torch

print("Checking cube_to_dir face assignments:")
print("=" * 60)

def cube_to_dir_fork(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -x, -y
    elif s == 1: rx, ry, rz = -torch.ones_like(x), x, -y
    elif s == 2: rx, ry, rz = x, y, torch.ones_like(x)
    elif s == 3: rx, ry, rz = x, -y, -torch.ones_like(x)
    elif s == 4: rx, ry, rz = x, torch.ones_like(x), -y
    elif s == 5: rx, ry, rz = -x, -torch.ones_like(x), -y
    return torch.stack((rx, ry, rz), dim=-1)

def cube_to_dir_official(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -x, -y
    elif s == 1: rx, ry, rz = -torch.ones_like(x), x, -y
    elif s == 2: rx, ry, rz = x, y, torch.ones_like(x)
    elif s == 3: rx, ry, rz = x, -y, -torch.ones_like(x)
    elif s == 4: rx, ry, rz = x, torch.ones_like(x), -y
    elif s == 5: rx, ry, rz = -x, -torch.ones_like(x), -y
    return torch.stack((rx, ry, rz), dim=-1)

# Check all faces
for s in range(6):
    x = torch.tensor([0.0])
    y = torch.tensor([0.0])
    d_fork = cube_to_dir_fork(s, x, y)[0]
    d_off = cube_to_dir_official(s, x, y)[0]
    
    # Determine dominant axis
    arr = d_fork.tolist()
    if arr[0] > 0.5: direction = "+X"
    elif arr[0] < -0.5: direction = "-X"
    elif arr[1] > 0.5: direction = "+Y"
    elif arr[1] < -0.5: direction = "-Y"
    elif arr[2] > 0.5: direction = "+Z"
    else: direction = "-Z"
    
    match = "MATCH" if torch.allclose(d_fork, d_off) else "MISMATCH"
    print(f"Face {s}: fork={arr}, direction={direction} [{match}]")

print("\nConclusion: Both use the SAME conventions (the code is identical).")
print("\nFace convention mapping:")
print("  Face 0: +X direction")
print("  Face 1: -X direction") 
print("  Face 2: +Z direction (NOT +Y!)")
print("  Face 3: -Z direction")
print("  Face 4: +Y direction (NOT +Z!)")
print("  Face 5: -Y direction")