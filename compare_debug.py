#!/usr/bin/env python3
"""
Compare debug outputs between DRTK (fork) and nvdiffrast (official) runs.

Usage:
    python compare_debug.py
"""

import numpy as np
import os

FORK_DIR = "test_output/debug"
OFFICIAL_DIR = "test_output_official/debug"


def load_npy(name, fork_dir=FORK_DIR, official_dir=OFFICIAL_DIR):
    """Load .npy from both directories and compare."""
    fork_path = os.path.join(fork_dir, name)
    official_path = os.path.join(official_dir, name)
    
    fork_data = np.load(fork_path) if os.path.exists(fork_path) else None
    official_data = np.load(official_path) if os.path.exists(official_path) else None
    
    return fork_data, official_data


def compare_tensor(name, fork_data, official_data, detailed=False):
    """Compare two tensors and print stats."""
    if fork_data is None or official_data is None:
        print(f"  {name}: MISSING (fork={fork_data is not None}, official={official_data is not None})")
        return
    
    fork_shape = fork_data.shape
    official_shape = official_data.shape
    
    if fork_shape != official_shape:
        print(f"  {name}: SHAPE MISMATCH fork={fork_shape} official={official_shape}")
        return
    
    # Shape matches, compare values
    if np.allclose(fork_data, official_data, rtol=1e-5, atol=1e-7):
        print(f"  {name}: ✓ IDENTICAL (shapes match, values within tolerance)")
        return
    
    # Different values
    diff = fork_data - official_data
    abs_diff = np.abs(diff)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    
    print(f"  {name}: ✗ DIFFERENT")
    print(f"    Fork:     shape={list(fork_shape)} min={fork_data.min():.6f} max={fork_data.max():.6f} mean={fork_data.mean():.6f} std={fork_data.std():.6f}")
    print(f"    Official: shape={list(official_shape)} min={official_data.min():.6f} max={official_data.max():.6f} mean={official_data.mean():.6f} std={official_data.std():.6f}")
    print(f"    Diff:     max_abs={max_diff:.6e} mean_abs={mean_diff:.6e}")
    
    if detailed and diff.size <= 100:
        print(f"    Fork values:     {fork_data.flatten()[:10]}")
        print(f"    Official values: {official_data.flatten()[:10]}")


def main():
    print("="*80)
    print("PIPELINE STAGE COMPARISON (P1-P10)")
    print("="*80)
    
    # Pipeline stages - should be IDENTICAL with same seed
    pipeline_files = [
        ("01_P1_ss_noise.npy", "Sparse structure input noise (should match with same seed)"),
        ("03_P2_ss_z_s_sampled.npy", "Sparse structure latent (CRITICAL - diverges here!)"),
        ("04_P3_ss_decoded.npy", "Decoded binary volume"),
        ("05_P4_ss_coords.npy", "Non-zero voxel coordinates"),
        ("07_P5_shape_slat_noise_feats.npy", "Shape SLat noise features"),
        ("08_P5_shape_slat_noise_coords.npy", "Shape SLat noise coords"),
        ("09_P6_shape_slat_sampled.npy", "Sampled shape SLat"),
        ("10_P7_shape_slat_denormed.npy", "Denormalized shape SLat"),
        ("11_P8_decode_shape_slat_input.npy", "Decode input"),
        ("12_P9_decode_mesh_vertices.npy", "Decoded mesh vertices"),
        ("13_P9_decode_mesh_faces.npy", "Decoded mesh faces"),
        ("15_P10_final_mesh_vertices.npy", "Final mesh vertices"),
        ("16_P10_final_mesh_faces.npy", "Final mesh faces"),
    ]
    
    for filename, desc in pipeline_files:
        print(f"\n{filename}")
        print(f"  Description: {desc}")
        fork_data, official_data = load_npy(filename)
        compare_tensor(filename, fork_data, official_data)
    
    print("\n" + "="*80)
    print("RENDERING STAGE COMPARISON")
    print("="*80)
    
    # Rendering stages - cameras should be identical
    render_files = [
        ("Fork: 08_R8_vertices_input.npy", "NVR_R8_vertices_input.npy", "Mesh vertices input to renderer"),
    ]
    
    # Compare camera/projection matrices
    print("\nCamera matrices (should match):")
    fork_extr, off_extr = load_npy("04_R4_extrinsics.npy", FORK_DIR, OFFICIAL_DIR)
    compare_tensor("R4_extrinsics", fork_extr, off_extr)
    
    fork_intr, off_intr = load_npy("05_R5_intrinsics.npy", FORK_DIR, OFFICIAL_DIR)
    compare_tensor("R5_intrinsics", fork_intr, off_intr)
    
    # Compare rasterizer outputs
    print("\n" + "-"*80)
    print("RASTERIZER OUTPUTS (key divergence point for rendering)")
    print("-"*80)
    
    # Fork uses DRTK, official uses nvdiffrast
    fork_rast = np.load(os.path.join(FORK_DIR, "17_DRTK_rast_output.npy"))
    off_rast = np.load(os.path.join(OFFICIAL_DIR, "14_NVR_R14_rast_output.npy"))
    
    print("\n  Rasterizer output comparison:")
    print(f"    Fork (DRTK):     shape={list(fork_rast.shape)} coverage={(fork_rast[...,3]>0).sum()/fork_rast[...,3].size*100:.2f}%")
    print(f"                     depth_range=[{fork_rast[0,...,2][fork_rast[0,...,3]>0].min():.4f}, {fork_rast[0,...,2][fork_rast[0,...,3]>0].max():.4f}]")
    print(f"                     bary_u_mean={fork_rast[0,...,0][fork_rast[0,...,3]>0].mean():.4f}")
    print(f"                     bary_v_mean={fork_rast[0,...,1][fork_rast[0,...,3]>0].mean():.4f}")
    
    print(f"    Official (nvdif): shape={list(off_rast.shape)} coverage={(off_rast[...,3]>0).sum()/off_rast[...,3].size*100:.2f}%")
    print(f"                     depth_range=[{off_rast[0,...,2][off_rast[0,...,3]>0].min():.4f}, {off_rast[0,...,2][off_rast[0,...,3]>0].max():.4f}]")
    print(f"                     bary_u_mean={off_rast[0,...,0][off_rast[0,...,3]>0].mean():.4f}")
    print(f"                     bary_v_mean={off_rast[0,...,1][off_rast[0,...,3]>0].mean():.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("""
The key finding is that the pipeline produces DIFFERENT meshes:
  - Fork:    383K vertices, 699K faces
  - Official: 1.47M vertices, 3.14M faces

This difference originates from the SPARSE STRUCTURE SAMPLING stage (P2),
where the latent tensors differ despite using the same random seed.

POSSIBLE CAUSES:
1. Different PyTorch/CUDA versions affecting RNG determinism
2. Different model weight caches
3. Import order affecting global RNG state
4. Different torch.backends flags (cudnn, etc.)

NEXT STEPS:
1. Check torch.__version__ and CUDA version in both environments
2. Verify same model weights are loaded (check hash or file size)
3. Set all deterministic flags before importing anything:
   torch.use_deterministic_algorithms(True)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
4. Check if the same .cache/huggingface model files are used
""")


if __name__ == "__main__":
    main()