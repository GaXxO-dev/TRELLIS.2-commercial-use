"""
Pure PyTorch PBR Environment Lighting (Vectorized Implementation).

This module replaces nvdiffrec's EnvironmentLight with a pure PyTorch implementation
of split-sum PBR environment lighting. No nvdiffrast or nvdiffrec dependencies.

All operations are fully vectorized using PyTorch batch operations for GPU efficiency.
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a tensor, avoiding division by zero."""
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)


def reflect(incident: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Reflect incident vector around normal."""
    return incident - 2.0 * torch.sum(incident * normal, dim=-1, keepdim=True) * normal


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dot product, keeping dimensions."""
    return torch.sum(a * b, dim=-1, keepdim=True)


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Convert cube face coordinates to 3D direction vectors.
    
    Convention matches nvdiffrast/nvdiffrec:
    Face 0 (+X): direction (1, -x, -y) maps UV (x,y)
    Face 1 (-X): direction (-1, x, -y) maps UV (x,y)
    Face 2 (+Y): direction (x, y, 1) maps UV (x,y)
    Face 3 (-Y): direction (x, -y, -1) maps UV (x,y)
    Face 4 (+Z): direction (x, 1, -y) maps UV (x,y)
    Face 5 (-Z): direction (-x, -1, -y) maps UV (x,y)
    """
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -x, -y
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), x, -y
    elif s == 2:
        rx, ry, rz = x, y, torch.ones_like(x)
    elif s == 3:
        rx, ry, rz = x, -y, -torch.ones_like(x)
    elif s == 4:
        rx, ry, rz = x, torch.ones_like(x), -y
    elif s == 5:
        rx, ry, rz = -x, -torch.ones_like(x), -y
    return torch.stack((rx, ry, rz), dim=-1)


def dir_to_cube_face_and_uv(directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert 3D direction vectors to cube face index and local UV coordinates.
    
    Fully vectorized - no Python loops over faces.
    
    Convention matches cube_to_dir:
    Face 0 (+X): cube_to_dir(0, x, y) = (1, -x, -y) -> direction = (1, -u, -v)/||.||
    Face 1 (-X): cube_to_dir(1, x, y) = (-1, x, -y) -> direction = (-1, u, -v)/||.||
    Face 2 (+Z): cube_to_dir(2, x, y) = (x, y, 1) -> direction = (u, v, 1)/||.||
    Face 3 (-Z): cube_to_dir(3, x, y) = (x, -y, -1) -> direction = (u, -v, -1)/||.||
    Face 4 (+Y): cube_to_dir(4, x, y) = (x, 1, -y) -> direction = (u, 1, -v)/||.||
    Face 5 (-Y): cube_to_dir(5, x, y) = (-x, -1, -y) -> direction = (-u, -1, -v)/||.||
    
    Args:
        directions: [..., 3] normalized direction vectors (x, y, z) = (dx, dy, dz)
        
    Returns:
        faces: [...] int64 face indices (0-5)
        u: [...] float local u coordinate [0, 1]
        v: [...] float local v coordinate [0, 1]
    """
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    
    abs_x = torch.abs(x)
    abs_y = torch.abs(y)
    abs_z = torch.abs(z)
    
    # Determine which axis is dominant
    # Face assignment: 0=+X, 1=-X, 2=+Z, 3=-Z, 4=+Y, 5=-Y
    # This matches cube_to_dir where s=2 gives +Z direction and s=4 gives +Y direction
    is_x_major = (abs_x >= abs_y) & (abs_x >= abs_z)
    is_z_major = (~is_x_major) & (abs_z >= abs_y)
    is_y_major = ~is_x_major & ~is_z_major
    
    # Compute UV coordinates for each face projection
    # These are inverses of cube_to_dir mappings
    
    # For X-major faces (0: +X when x>0, 1: -X when x<0)
    # cube_to_dir(0, x, y) = (1, -x, -y) so u=-y, v=-... 
    # From direction (dx, dy, dz): y_input = -dy/|dx|, z_input (actually unused) from -y = -dz
    # Actually: direction = normalize(1, -u_input, -v_input)
    # So: u_input = -dy/|dx|, v_input = -dz/|dx|
    # UV coords: u = (u_input+1)/2 = (-dy/|dx|+1)/2, v = (v_input+1)/2 = (-dz/|dx|+1)/2
    
    # For +X face (dx > 0): u = (-dy/|dx|+1)/2, v = (-dz/|dx|+1)/2
    # For -X face (dx < 0): cube_to_dir(1, u, v) = (-1, u, -v)
    #                       so: u_input = dy/|dx|, v_input = -dz/|dx|
    #                       UV: u = (dy/|dx|+1)/2, v = (-dz/|dx|+1)/2
    
    u_for_pos_x = (-y / abs_x + 1) * 0.5  # u = (-dy/|dx|+1)/2
    v_for_pos_x = (-z / abs_x + 1) * 0.5   # v = (-dz/|dx|+1)/2
    u_for_neg_x = (y / abs_x + 1) * 0.5    # u = (dy/|dx|+1)/2
    v_for_neg_x = (-z / abs_x + 1) * 0.5   # v = (-dz/|dx|+1)/2
    
    u_x = torch.where(x > 0, u_for_pos_x, u_for_neg_x)
    v_x = torch.where(x > 0, v_for_pos_x, v_for_neg_x)
    
    # For Z-major faces (2: +Z when z>0, 3: -Z when z<0)
    # cube_to_dir(2, u, v) = (u, v, 1) so direction = normalize(dx, dy, dz) with dz > 0
    # u_input = dx/|dz|, v_input = dy/|dz|
    # UV: u = (dx/|dz|+1)/2, v = (dy/|dz|+1)/2
    #
    # cube_to_dir(3, u, v) = (u, -v, -1) so direction with dz < 0
    # u_input = dx/|dz|, v_input = -dy/|dz|
    # UV: u = (dx/|dz|+1)/2, v = (-dy/|dz|+1)/2
    
    u_for_pos_z = (x / abs_z + 1) * 0.5
    v_for_pos_z = (y / abs_z + 1) * 0.5
    u_for_neg_z = (x / abs_z + 1) * 0.5
    v_for_neg_z = (-y / abs_z + 1) * 0.5
    
    u_z = torch.where(z > 0, u_for_pos_z, u_for_neg_z)
    v_z = torch.where(z > 0, v_for_pos_z, v_for_neg_z)
    
    # For Y-major faces (4: +Y when y>0, 5: -Y when y<0)
    # cube_to_dir(4, u, v) = (u, 1, -v) so direction with dy > 0
    # u_input = dx/|dy|, v_input = -dz/|dy|
    # UV: u = (dx/|dy|+1)/2, v = (-dz/|dy|+1)/2
    #
    # cube_to_dir(5, u, v) = (-u, -1, -v) so direction with dy < 0
    # u_input = -dx/|dy|, v_input = -dz/|dy|
    # UV: u = (-dx/|dy|+1)/2, v = (-dz/|dy|+1)/2
    
    u_for_pos_y = (x / abs_y + 1) * 0.5
    v_for_pos_y = (-z / abs_y + 1) * 0.5
    u_for_neg_y = (-x / abs_y + 1) * 0.5
    v_for_neg_y = (-z / abs_y + 1) * 0.5
    
    u_y = torch.where(y > 0, u_for_pos_y, u_for_neg_y)
    v_y = torch.where(y > 0, v_for_pos_y, v_for_neg_y)
    
    # Select UV based on dominant axis
    u = torch.where(is_x_major, u_x, torch.where(is_z_major, u_z, u_y))
    v = torch.where(is_x_major, v_x, torch.where(is_z_major, v_z, v_y))
    
    # Compute face indices: 0=+X, 1=-X, 2=+Z, 3=-Z, 4=+Y, 5=-Y
    faces = torch.where(
        is_x_major,
        torch.where(x > 0, torch.tensor(0, device=directions.device), torch.tensor(1, device=directions.device)),
        torch.where(
            is_z_major,
            torch.where(z > 0, torch.tensor(2, device=directions.device), torch.tensor(3, device=directions.device)),
            torch.where(y > 0, torch.tensor(4, device=directions.device), torch.tensor(5, device=directions.device))
        )
    )
    
    return faces.long(), u, v
    
    # Compute face indices: 0=+X, 1=-X, 2=+Z, 3=-Z, 4=+Y, 5=-Y
    faces = torch.where(
        is_x_major,
        torch.where(x > 0, torch.tensor(0, device=directions.device), torch.tensor(1, device=directions.device)),
        torch.where(
            is_z_major,
            torch.where(z > 0, torch.tensor(2, device=directions.device), torch.tensor(3, device=directions.device)),
            torch.where(y > 0, torch.tensor(4, device=directions.device), torch.tensor(5, device=directions.device))
        )
    )
    
    return faces.long(), u, v
    
    # Compute face indices
    faces = torch.where(
        is_x_major,
        torch.where(x > 0, torch.tensor(0, device=directions.device), torch.tensor(1, device=directions.device)),
        torch.where(
            is_y_major,
            torch.where(y > 0, torch.tensor(2, device=directions.device), torch.tensor(3, device=directions.device)),
            torch.where(z > 0, torch.tensor(4, device=directions.device), torch.tensor(5, device=directions.device))
        )
    )
    
    return faces.long(), u, v


def sample_cubemap(cubemap: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Sample a cubemap given direction vectors - fully vectorized.
    
    Args:
        cubemap: [6, H, W, C] cubemap faces (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
        directions: [..., 3] normalized direction vectors
        
    Returns:
        [..., C] sampled colors
    """
    original_shape = directions.shape[:-1]
    C = cubemap.shape[-1]
    H, W = cubemap.shape[1], cubemap.shape[2]
    
    directions_flat = directions.reshape(-1, 3)
    N = directions_flat.shape[0]
    
    faces, u, v = dir_to_cube_face_and_uv(directions_flat)
    
    # Convert to grid_sample coordinates [-1, 1]
    grid_x = u * 2 - 1
    grid_y = v * 2 - 1
    
    # Process all 6 faces in parallel
    # For each face, we need to sample and then select based on which face each pixel belongs to
    result = torch.zeros(N, C, dtype=cubemap.dtype, device=cubemap.device)
    
    # Stack all 6 faces as [6, C, H, W]
    cubemap_chw = cubemap.permute(0, 3, 1, 2)  # [6, C, H, W]
    
    for face_idx in range(6):
        mask = faces == face_idx
        if not mask.any():
            continue
        
        # Get directions for this face
        face_dirs = directions_flat[mask]
        face_n = face_dirs.shape[0]
        
        # Get UV for this face
        face_grid_x = grid_x[mask]
        face_grid_y = grid_y[mask]
        
        # Create grid for grid_sample: [1, N, 1, 2]
        face_grid = torch.stack([face_grid_x, face_grid_y], dim=-1).unsqueeze(0).unsqueeze(2)
        
        # Sample from this face
        face_chw = cubemap_chw[face_idx:face_idx+1]  # [1, C, H, W]
        sampled = F.grid_sample(face_chw, face_grid, mode='bilinear', padding_mode='border', align_corners=True)
        # sampled: [1, C, N, 1]
        
        result[mask] = sampled[0, :, :, 0].permute(1, 0)
    
    return result.reshape(*original_shape, C)


def sample_cubemap_batch(cubemap: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Sample a cubemap - batch optimized version.
    
    Processes all 6 faces in a single batched grid_sample call by batching
    directions by face and concatenating results.
    
    Args:
        cubemap: [6, H, W, C] cubemap faces
        directions: [B, ..., 3] normalized direction vectors
        
    Returns:
        [B, ..., C] sampled colors
    """
    original_shape = directions.shape[:-1]
    C = cubemap.shape[-1]
    H, W = cubemap.shape[1], cubemap.shape[2]
    
    directions_flat = directions.reshape(-1, 3)
    N = directions_flat.shape[0]
    
    faces, u, v = dir_to_cube_face_and_uv(directions_flat)
    
    # Convert to grid_sample coordinates
    grid_x = u * 2 - 1
    grid_y = v * 2 - 1
    
    # Prepare cubemap as [1, 6*C, H, W] so we can do a single grid_sample
    # Actually, we need to handle faces separately since each direction maps to one face
    # But we can batch by face
    
    cubemap_chw = cubemap.permute(0, 3, 1, 2)  # [6, C, H, W]
    
    # Build result by face
    result = torch.zeros(N, C, dtype=cubemap.dtype, device=cubemap.device)
    
    for face_idx in range(6):
        mask = faces == face_idx
        count = mask.sum().item()
        if count == 0:
            continue
        
        # Indices in result array
        idx = torch.where(mask)[0]
        
        # Grid for this face's directions
        fx = grid_x[mask]
        fy = grid_y[mask]
        
        # Reshape grid for grid_sample: [1, count, 1, 2]
        face_grid = torch.stack([fx, fy], dim=-1).unsqueeze(0).unsqueeze(2)
        
        # Sample
        face_chw = cubemap_chw[face_idx:face_idx+1]
        sampled = F.grid_sample(face_chw.expand(count, -1, -1, -1), 
                                face_grid.expand(count, -1, -1, -1),
                                mode='bilinear', padding_mode='border', align_corners=True)
        # sampled: [count, C, 1, 1]
        
        result[idx] = sampled[:, :, 0, 0].T
    
    return result.reshape(*original_shape, C)


def sample_cubemap_mip(cubemap_mips: List[torch.Tensor], directions: torch.Tensor, mip_level: torch.Tensor) -> torch.Tensor:
    """Sample a mipped cubemap given direction vectors and mip level.
    
    Args:
        cubemap_mips: List of [6, H, W, C] cubemap faces at different mip levels
        directions: [..., 3] normalized direction vectors
        mip_level: [...] float mip level
        
    Returns:
        [..., C] sampled colors
    """
    num_mips = len(cubemap_mips)
    
    if num_mips == 1:
        return sample_cubemap(cubemap_mips[0], directions)
    
    original_shape = directions.shape[:-1]
    C = cubemap_mips[0].shape[-1]
    directions_flat = directions.reshape(-1, 3)
    N = directions_flat.shape[0]
    mip_level_flat = mip_level.reshape(-1)
    
    mip_level_clamped = torch.clamp(mip_level_flat, 0, num_mips - 1)
    mip_low = mip_level_clamped.floor().long()
    mip_high = (mip_low + 1).clamp(max=num_mips - 1)
    t = (mip_level_clamped - mip_low.float()).unsqueeze(-1)
    
    result = torch.zeros(N, C, dtype=cubemap_mips[0].dtype, device=directions.device)
    
    for mip_idx in range(num_mips):
        mask = (mip_low == mip_idx) | (mip_high == mip_idx)
        if not mask.any():
            continue
        
        idx = torch.where(mask)[0]
        dirs_this = directions_flat[idx]
        mip_low_this = mip_low[idx]
        mip_high_this = mip_high[idx]
        
        low_mip_idx = mip_low_this[0].item()
        high_mip_idx = mip_high_this[0].item()
        
        if low_mip_idx == high_mip_idx:
            sampled = sample_cubemap(cubemap_mips[low_mip_idx], dirs_this)
            result[idx] = sampled
        else:
            sampled_low = sample_cubemap(cubemap_mips[low_mip_idx], dirs_this)
            sampled_high = sample_cubemap(cubemap_mips[high_mip_idx], dirs_this)
            t_this = t[idx]
            result[idx] = torch.lerp(sampled_low, sampled_high, t_this)
    
    return result.reshape(*original_shape, C)


def avg_pool_cubemap(cubemap: torch.Tensor) -> torch.Tensor:
    """Average pool a cubemap by 2x."""
    cubemap_chw = cubemap.permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(cubemap_chw, 2)
    return pooled.permute(0, 2, 3, 1)


def build_cubemap_mips(cubemap: torch.Tensor, min_res: int = 16) -> List[torch.Tensor]:
    """Build mipmap chain from cubemap."""
    mips = [cubemap]
    while mips[-1].shape[1] > min_res:
        mips.append(avg_pool_cubemap(mips[-1]))
    return mips


def compute_diffuse_cubemap_vectorized(cubemap: torch.Tensor, num_samples: int = 1024) -> torch.Tensor:
    """Compute diffuse irradiance cubemap via vectorized Monte Carlo integration.
    
    Args:
        cubemap: [6, H, W, C] environment cubemap
        num_samples: Number of samples per direction
        
    Returns:
        [6, H, W, C] diffuse irradiance cubemap
    """
    device = cubemap.device
    dtype = cubemap.dtype
    C = cubemap.shape[-1]
    H, W = cubemap.shape[1], cubemap.shape[2]
    
    # Generate all texel directions for all 6 faces
    result = torch.zeros(6, H, W, C, dtype=dtype, device=device)
    
    # Use importance sampling for hemisphere - sample directions in tangent space
    # then transform to world space per face
    phi = torch.linspace(0, 2 * math.pi, int(math.sqrt(num_samples)) + 1, device=device)[:-1]
    cos_theta = torch.linspace(0, 1, int(math.sqrt(num_samples)) + 1, device=device)[:-1]
    phi_grid, cos_theta_grid = torch.meshgrid(phi, cos_theta, indexing='ij')
    
    # Cosine-weighted hemisphere sampling in tangent space
    sin_theta_grid = torch.sqrt(1 - cos_theta_grid ** 2)
    sample_dirs_tangent = torch.stack([
        sin_theta_grid * torch.cos(phi_grid),
        sin_theta_grid * torch.sin(phi_grid),
        cos_theta_grid
    ], dim=-1)  # [N_phi, N_theta, 3]
    sample_weights = cos_theta_grid  # Cosine weighting
    sample_dirs_tangent = sample_dirs_tangent.reshape(-1, 3)  # [N, 3]
    sample_weights = sample_weights.reshape(-1)  # [N]
    N_samples = sample_dirs_tangent.shape[0]
    
    for face_idx in range(6):
        # Generate texel positions for this cubemap face
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device),
            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device),
            indexing='ij'
        )
        
        # Normal direction for each texel
        normals = safe_normalize(cube_to_dir(face_idx, gx.reshape(-1), gy.reshape(-1)))
        normals = normals.reshape(H * W, 3)  # [H*W, 3]
        
        # Build tangent frame for each texel
        up = torch.zeros_like(normals)
        up[:, 2] = 1.0
        parallel_to_z = torch.abs(normals[:, 2]) > 0.999
        up[parallel_to_z] = torch.tensor([1.0, 0.0, 0.0], device=device)
        
        tangent = safe_normalize(torch.linalg.cross(up, normals, dim=-1))
        bitangent = torch.linalg.cross(normals, tangent, dim=-1)
        
        # Transform all tangent-space samples to world space for each texel
        # sample_dirs_tangent: [N, 3]
        # normals, tangent, bitangent: [H*W, 3]
        # Result: [H*W, N, 3]
        
        # Use einsum for batch matrix multiplication
        # world_dirs = tangent @ sample_dirs[..., 0] + bitangent @ sample_dirs[..., 1] + normals @ sample_dirs[..., 2]
        world_dirs = (
            sample_dirs_tangent[..., 0:1] * tangent.unsqueeze(1) +
            sample_dirs_tangent[..., 1:2] * bitangent.unsqueeze(1) +
            sample_dirs_tangent[..., 2:3] * normals.unsqueeze(1)
        )
        world_dirs = safe_normalize(world_dirs)  # [H*W, N, 3]
        
        # Sample the environment map for all these directions
        # world_dirs: [H*W, N, 3] -> need to flatten, sample, then reshape
        world_dirs_flat = world_dirs.reshape(-1, 3)  # [H*W * N, 3]
        radiance_flat = sample_cubemap(cubemap, world_dirs_flat)  # [H*W * N, C]
        radiance = radiance_flat.reshape(H * W, N_samples, C)  # [H*W, N, C]
        
        # Weight by cosine and average
        irradiance = (radiance * sample_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # [H*W, C]
        irradiance = irradiance * (math.pi / N_samples)
        
        result[face_idx] = irradiance.reshape(H, W, C)
    
    return result


def compute_specular_prefilter_vectorized(cubemap: torch.Tensor, roughness: float, num_samples: int = 128) -> torch.Tensor:
    """Compute prefiltered specular cubemap via vectorized GGX importance sampling.
    
    Args:
        cubemap: [6, H, W, C] environment cubemap
        roughness: Roughness value [0, 1]
        num_samples: Number of samples per direction
        
    Returns:
        [6, H, W, C] prefiltered cubemap
    """
    device = cubemap.device
    dtype = cubemap.dtype
    C = cubemap.shape[-1]
    H, W = cubemap.shape[1], cubemap.shape[2]
    
    result = torch.zeros(6, H, W, C, dtype=dtype, device=device)
    
    roughness = max(roughness, 1e-4)
    a = roughness * roughness
    a2 = a * a
    
    # GGX importance sampling
    phi = torch.linspace(0, 2 * math.pi, int(math.sqrt(num_samples)) + 1, device=device)[:-1]
    u = torch.linspace(0, 1, int(math.sqrt(num_samples)) + 1, device=device)[:-1]
    phi_grid, u_grid = torch.meshgrid(phi, u, indexing='ij')
    
    # GGX distribution importance sampling
    cos_theta = torch.sqrt((1 - u_grid) / (1 + (a2 - 1) * u_grid))
    sin_theta = torch.sqrt(1 - cos_theta ** 2)
    
    # Sample directions in tangent space (half-vector)
    h_tangent = torch.stack([
        sin_theta * torch.cos(phi_grid),
        sin_theta * torch.sin(phi_grid),
        cos_theta
    ], dim=-1)  # [N_phi, N_u, 3]
    
    h_tangent = h_tangent.reshape(-1, 3)  # [N, 3]
    N_samples = h_tangent.shape[0]
    
    # Weight by NdotH (cosine of half-vector with normal)
    weights = cos_theta.reshape(-1)  # [N]
    
    for face_idx in range(6):
        # Generate texel positions for this cubemap face
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device),
            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device),
            indexing='ij'
        )
        
        # Normal direction for each texel (assumes N = V for prefiltering)
        normals = safe_normalize(cube_to_dir(face_idx, gx.reshape(-1), gy.reshape(-1)))
        normals = normals.reshape(H * W, 3)  # [H*W, 3]
        
        # Build tangent frame
        up = torch.zeros_like(normals)
        up[:, 2] = 1.0
        parallel_to_z = torch.abs(normals[:, 2]) > 0.999
        up[parallel_to_z] = torch.tensor([1.0, 0.0, 0.0], device=device)
        
        tangent = safe_normalize(torch.linalg.cross(up, normals, dim=-1))
        bitangent = torch.linalg.cross(normals, tangent, dim=-1)
        
        # Transform half-vectors to world space
        h_world = (
            h_tangent[..., 0:1] * tangent.unsqueeze(1) +
            h_tangent[..., 1:2] * bitangent.unsqueeze(1) +
            h_tangent[..., 2:3] * normals.unsqueeze(1)
        )
        h_world = safe_normalize(h_world)  # [H*W, N, 3]
        
        # For prefiltering with N=V assumption, light direction = reflect(N, H) = 2*dot(N,H)*H - N
        NdotH = (normals.unsqueeze(1) * h_world).sum(dim=-1, keepdim=True)  # [H*W, N, 1]
        light_dir = safe_normalize(2 * NdotH * h_world - normals.unsqueeze(1))
        
        # Weight by NdotH (cosine-weighted sampling)
        # Note: with GGX importance sampling, the PDF cancels most terms
        
        # Sample environment map
        light_dirs_flat = light_dir.reshape(-1, 3)
        radiance_flat = sample_cubemap(cubemap, light_dirs_flat)
        radiance = radiance_flat.reshape(H * W, N_samples, C)
        
        # Weighted average
        prefiltered = (radiance * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        weight_sum = weights.sum()
        prefiltered = prefiltered / weight_sum
        
        result[face_idx] = prefiltered.reshape(H, W, C)
    
    return result


def load_bsdf_lut(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load the precomputed BSDF LUT for split-sum approximation."""
    lut_path = os.path.join(os.path.dirname(__file__), 'pbr_envmap_data', 'bsdf_256_256.bin')
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f"BSDF LUT not found at {lut_path}")
    
    lut_data = np.fromfile(lut_path, dtype=np.float32).reshape(1, 256, 256, 2)
    return torch.tensor(lut_data, dtype=dtype, device=device)


class PBREnvironmentLight(torch.nn.Module):
    """PBR Environment Light using split-sum approximation.
    
    This class provides image-based lighting (IBL) for PBR materials using the
    split-sum approximation. All computations are vectorized for GPU efficiency.
    """
    
    LIGHT_MIN_RES = 16
    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5
    
    def __init__(self, base: torch.Tensor):
        """Initialize the environment light.
        
        Args:
            base: [6, H, W, C] cubemap of the environment
        """
        super().__init__()
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.mtx = None
        
        self.diffuse: Optional[torch.Tensor] = None
        self.specular: List[torch.Tensor] = []
        self._fg_lut: Optional[torch.Tensor] = None
    
    def xfm(self, mtx: torch.Tensor):
        """Set transformation matrix for light rotation."""
        self.mtx = mtx
    
    def clone(self) -> 'PBREnvironmentLight':
        """Clone the environment light."""
        return PBREnvironmentLight(self.base.clone().detach())
    
    def clamp_(self, min: Optional[float] = None, max: Optional[float] = None):
        """Clamp the base environment map values."""
        self.base.clamp_(min, max)
    
    def get_mip(self, roughness: torch.Tensor) -> torch.Tensor:
        """Get mip level for given roughness."""
        num_specular = len(self.specular)
        
        low_roughness_mip = (
            torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) 
            - self.MIN_ROUGHNESS
        ) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (num_specular - 2)
        
        high_roughness_mip = (
            torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) 
            - self.MAX_ROUGHNESS
        ) / (1.0 - self.MAX_ROUGHNESS) + num_specular - 2
        
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            low_roughness_mip,
            high_roughness_mip
        )
    
    def build_mips(self, cutoff: float = 0.99, diffuse_samples: int = 1024, specular_samples: int = 128):
        """Build the mipmap chain and prefilter the environment.
        
        Uses vectorized Monte Carlo integration for efficiency.
        
        Args:
            cutoff: Cutoff value (unused, kept for compatibility)
            diffuse_samples: Number of samples for diffuse convolution
            specular_samples: Number of samples for specular prefiltering
        """
        # Build mipmap chain
        self.specular = build_cubemap_mips(self.base, self.LIGHT_MIN_RES)
        
        # Compute diffuse irradiance at lowest resolution (matching nvdiffrec behavior)
        self.diffuse = compute_diffuse_cubemap_vectorized(
            self.specular[-1].data, 
            num_samples=diffuse_samples
        )
        
        # Prefilter specular at each mip level
        num_levels = len(self.specular)
        for idx in range(num_levels - 1):
            roughness = (idx / max(num_levels - 2, 1)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = compute_specular_prefilter_vectorized(
                self.specular[idx].data, 
                roughness, 
                num_samples=specular_samples
            )
        
        # Last mip level has roughness = 1.0
        self.specular[-1] = compute_specular_prefilter_vectorized(
            self.specular[-1].data, 
            1.0, 
            num_samples=specular_samples
        )
        
        self._fg_lut = load_bsdf_lut(self.base.device, self.base.dtype)
    
    def shade(
        self,
        gb_pos: torch.Tensor,
        gb_normal: torch.Tensor,
        kd: torch.Tensor,
        ks: torch.Tensor,
        view_pos: torch.Tensor,
        specular: bool = True
    ) -> torch.Tensor:
        """Shade surface points using split-sum PBR.
        
        Args:
            gb_pos: [1, H, W, 3] world-space surface positions
            gb_normal: [1, H, W, 3] world-space surface normals (normalized)
            kd: [1, H, W, 3] diffuse albedo (base color)
            ks: [1, H, W, 3] ORM (occlusion in R, roughness in G, metallic in B)
            view_pos: [1, H, W, 3] or [1, 1, 1, 3] view/camera position
            specular: Whether to compute specular term
            
        Returns:
            [1, H, W, 3] shaded color
        """
        wo = safe_normalize(view_pos - gb_pos)
        
        if specular:
            roughness = ks[..., 1:2]
            metallic = ks[..., 2:3]
            spec_col = (1.0 - metallic) * 0.04 + kd * metallic
            diff_col = kd * (1.0 - metallic)
        else:
            diff_col = kd
        
        reflvec = safe_normalize(reflect(wo, gb_normal))
        nrmvec = gb_normal
        
        if self.mtx is not None:
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device=gb_pos.device)
            reflvec_flat = reflvec.view(reflvec.shape[0], -1, reflvec.shape[-1])
            reflvec = torch.bmm(reflvec_flat, mtx.T.unsqueeze(0)).view(*reflvec.shape)
            
            nrmvec_flat = nrmvec.view(nrmvec.shape[0], -1, nrmvec.shape[-1])
            nrmvec = torch.bmm(nrmvec_flat, mtx.T.unsqueeze(0)).view(*nrmvec.shape)
        
        # Diffuse lookup
        diffuse = sample_cubemap(self.diffuse, nrmvec)
        shaded_col = diffuse * diff_col
        
        if specular:
            NdotV = torch.clamp(dot(wo, gb_normal), min=1e-4)
            roughness_clamped = torch.clamp(roughness, 0.0, 1.0)
            fg_uv = torch.cat([NdotV, roughness_clamped], dim=-1)
            
            fg_lookup = F.grid_sample(
                self._fg_lut.permute(0, 3, 1, 2),
                fg_uv * 2 - 1,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).permute(0, 2, 3, 1)
            
            miplevel = self.get_mip(roughness)
            spec = sample_cubemap_mip(self.specular, reflvec, miplevel[..., 0])
            
            # Compute aggregate lighting (split-sum approximation)
            reflectance = spec_col * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
            shaded_col = shaded_col + spec * reflectance
        
        return shaded_col * (1.0 - ks[..., 0:1])

    def regularizer(self) -> torch.Tensor:
        """Regularization term for the environment map."""
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))