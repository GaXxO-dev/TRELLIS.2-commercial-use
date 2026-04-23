#!/usr/bin/env python3
"""Test cube_to_dir <-> dir_to_cube_face_and_uv consistency."""
import torch
import sys
sys.path.insert(0, '/home/gaxxo/Documents/AI/TRELLIS.2-commercial-use')

from trellis2.renderers.pbr_envmap import cube_to_dir, dir_to_cube_face_and_uv

print("Testing cube_to_dir <-> dir_to_cube_face_and_uv consistency:")
print("=" * 70)

# Test that the functions are inverses
errors_found = []
for s in range(6):
    for u_test in [-0.8, -0.3, 0.0, 0.3, 0.8]:
        for v_test in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            x = torch.tensor([u_test])
            y = torch.tensor([v_test])
            
            # Forward: UV -> direction
            direction = cube_to_dir(s, x, y)  # [1, 3]
            
            # Inverse: direction -> face, UV
            faces_out, u_out, v_out = dir_to_cube_face_and_uv(direction)
            
            # Convert back
            u_back = u_out[0].item() * 2 - 1
            v_back = v_out[0].item() * 2 - 1
            direction_back = cube_to_dir(faces_out[0].item(), torch.tensor([u_back]), torch.tensor([v_back]))
            
            error = (direction - direction_back).abs().max().item()
            
            if error > 0.001 or faces_out[0].item() != s:
                errors_found.append({
                    'face': s,
                    'u': u_test,
                    'v': v_test,
                    'direction': direction[0].tolist(),
                    'recovered_face': faces_out[0].item(),
                    'u_out': u_out[0].item(),
                    'v_out': v_out[0].item(),
                    'back_dir': direction_back[0].tolist(),
                    'error': error
                })

if errors_found:
    print(f"Found {len(errors_found)} errors:")
    for err in errors_found[:5]:  # Show first 5
        print(f"Face {err['face']}: UV ({err['u']:.2f}, {err['v']:.2f})")
        print(f"  Direction: {err['direction']}")
        print(f"  Recovered: face={err['recovered_face']}, UV=({err['u_out']:.4f}, {err['v_out']:.4f})")
        print(f"  Back-dir: {err['back_dir']}")
        print(f"  Error: {err['error']:.6f}")
        print()
else:
    print("All roundtrip tests passed!")

# Also test that directions map to correct faces
print("\nTesting face assignment:")
for s in range(6):
    # At center of face (u=0, v=0), direction should point along face axis
    direction = cube_to_dir(s, torch.tensor([0.0]), torch.tensor([0.0]))
    print(f"Face {s}: direction = {direction[0].tolist()}")