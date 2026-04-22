"""
DRTK compatibility layer providing nvdiffrast-like API semantics.

This module wraps DRTK functions to provide an interface similar to nvdiffrast,
making migration easier while using DRTK's MIT-licensed implementation.

Key API differences:
- nvdiffrast uses clip-space coordinates [N, V, 4], DRTK uses pixel-space [N, V, 3]
- nvdiffrast rasterize returns (rast, rast_db), DRTK returns index_img + render() gives (depth, bary)
- nvdiffrast interpolate takes rast, DRTK interpolate takes (index_img, bary_img)
- nvdiffrast texture uses derivatives, DRTK mipmap_grid_sample uses Jacobian
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import functools
from .debug_utils import is_debug_enabled, dbg_tensor, dbg_value, dbg_rast_stats, next_step, get_debug_dir
import os
import numpy as np


def build_mipmap(tex: torch.Tensor, max_levels: int = 12) -> List[torch.Tensor]:
    """Build mipmap pyramid from texture tensor.
    
    Args:
        tex: Texture tensor [N, C, H, W]
        max_levels: Maximum mipmap levels to generate
        
    Returns:
        List of mipmap levels, starting with original texture
    """
    mipmaps = [tex]
    h, w = tex.shape[-2], tex.shape[-1]
    for _ in range(1, max_levels):
        if h <= 1 and w <= 1:
            break
        tex = F.avg_pool2d(tex, 2)
        mipmaps.append(tex)
        h, w = (h + 1) // 2, (w + 1) // 2
    return mipmaps


def compute_uv_jacobian(uv: torch.Tensor, resolution: int) -> torch.Tensor:
    """Compute UV Jacobian for mipmap level selection.
    
    This computes the derivatives of UV coordinates with respect to pixel position,
    needed for DRTK's mipmap_grid_sample. Uses finite differences.
    
    Args:
        uv: UV coordinates [N, H, W, 2]
        resolution: Image resolution (used for scaling)
        
    Returns:
        Jacobian tensor [N, H, W, 2, 2]
    """
    n, h, w, _ = uv.shape
    
    # Compute gradients using finite differences
    # dudx, dudy, dvdx, dvdy
    dx = uv[:, :, 1:, :] - uv[:, :, :-1, :]  # [N, H, W-1, 2]
    dy = uv[:, 1:, :, :] - uv[:, :-1, :, :]  # [N, H-1, W, 2]
    
    # Pad to original size
    dx = F.pad(dx, (0, 0, 0, 1), mode='replicate')  # [N, H, W, 2]
    dy = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='replicate')  # [N, H, W, 2]
    
    # Construct jacobian [N, H, W, 2, 2]
    jacobian = torch.zeros(n, h, w, 2, 2, device=uv.device, dtype=uv.dtype)
    jacobian[..., 0, 0] = dx[..., 0]  # du/dx
    jacobian[..., 0, 1] = dy[..., 0]  # du/dy
    jacobian[..., 1, 0] = dx[..., 1]  # dv/dx
    jacobian[..., 1, 1] = dy[..., 1]  # dv/dy
    
    return jacobian


def intrinsics_to_camera_params(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: int,
    near: float = 0.1,
    far: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert OpenCV camera intrinsics/extrinsics to DRTK camera parameters.
    
    Args:
        intrinsics: [3, 3] OpenCV intrinsics matrix
        extrinsics: [4, 4] camera extrinsics matrix (world to camera)
        resolution: Image resolution (H=W assumed)
        near: Near plane (unused, for compatibility)
        far: Far plane (unused, for compatibility)
        
    Returns:
        campos: Camera position [3]
        camrot: Camera rotation matrix [3, 3]
        focal: Focal lengths [2, 2]
        princpt: Principal point [2]
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Camera position: -R^T @ t from extrinsics [R|t]
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    campos = -R.T @ t
    
    # Camera rotation
    camrot = R.T
    
    # Focal length matrix and principal point
    focal = torch.stack([
        torch.stack([fx, torch.zeros_like(fx)]),
        torch.stack([torch.zeros_like(fy), fy])
    ])
    princpt = torch.stack([cx, cy])
    
    return campos, camrot, focal, princpt


class DRTKContext:
    """Compatibility context mimicking nvdiffrast's RasterizeCudaContext.
    
    DRTK is stateless, but this provides a compatible interface.
    Caches rasterization outputs for use in interpolate().
    """
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._cache_index_img = None
        self._cache_bary = None
        self._cache_resolution = None
    
    def rasterize(
        self,
        vertices_clip: torch.Tensor,
        faces: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize mesh using DRTK, returning nvdiffrast-compatible output.
        
        Args:
            vertices_clip: Clip-space vertices [1, V, 4]
            faces: Face indices [F, 3]
            resolution: (H, W) output resolution
            
        Returns:
            rast: Rasterization output [1, H, W, 4] similar to nvdiffrast format
                - [..., 0:2]: barycentric coordinates (v, u)
                - [..., 2]: depth (z/w)
                - [..., 3]: triangle ID (1-indexed, 0 means background)
            rast_db: Barycentric derivatives [1, H, W, 4] (placeholder)
        """
        import drtk
        
        h, w = resolution
        
        if is_debug_enabled():
            step = next_step()
            dbg_tensor(step, "DRTK_input_vertices_clip", vertices_clip)
            dbg_value(step, "DRTK_input_resolution", resolution)
            dbg_tensor(step, "DRTK_input_faces", faces)
        
        w_clip = vertices_clip[..., 3].clamp(min=1e-8, max=1e8)
        x_ndc = vertices_clip[..., 0] / w_clip
        y_ndc = -vertices_clip[..., 1] / w_clip
        z_ndc = vertices_clip[..., 2] / w_clip
        
        x_pix = (x_ndc + 1) * 0.5 * w - 0.5
        y_pix = (h - 1) - ((y_ndc + 1) * 0.5 * h - 0.5)
        z_cam = vertices_clip[..., 3].clone()
        
        v_pix = torch.stack([x_pix, y_pix, z_cam], dim=-1)
        
        if is_debug_enabled():
            step = next_step()
            dbg_tensor(step, "DRTK_v_pix", v_pix)
        
        faces_int = faces.to(torch.int32) if faces.dtype != torch.int32 else faces
        
        index_img = drtk.rasterize(v_pix, faces_int, height=h, width=w)
        depth, bary = drtk.render(v_pix, faces_int, index_img)
        
        if is_debug_enabled():
            step = next_step()
            dbg_tensor(step, "DRTK_raw_index_img", index_img)
            dbg_tensor(step, "DRTK_raw_depth", depth)
            dbg_tensor(step, "DRTK_raw_bary", bary)
        
        # Ensure index_img has batch dimension [N, H, W]
        if index_img.dim() == 2:
            index_img = index_img.unsqueeze(0)
        
        # Cache for use in interpolate()
        self._cache_index_img = index_img
        self._cache_bary = bary
        self._cache_resolution = (h, w)
        
        batch_size = v_pix.shape[0]
        rast = torch.zeros(batch_size, h, w, 4, device=v_pix.device, dtype=torch.float32)
        
        # DRTK returns: depth [batch, H, W], bary [batch, 3, H, W]
        # bary dimensions are (w0, w1, w2) = weights for vertices 0, 1, 2
        # nvdiffrast uses (u, v) where u=weight for v1, v=weight for v2
        # So: u = bary[0, 1], v = bary[0, 2]
        rast[0, ..., 0] = bary[0, 1]  # u = weight for vertex 1
        rast[0, ..., 1] = bary[0, 2]  # v = weight for vertex 2
        rast[0, ..., 2] = depth[0]    # camera-space depth
        rast[0, ..., 3] = (index_img[0].float() + 1).float()
        
        rast_db = torch.zeros_like(rast)
        
        if is_debug_enabled():
            step = next_step()
            dbg_tensor(step, "DRTK_rast_output", rast)
            dbg_rast_stats(step, rast, "DRTK_rast")
            dbg_tensor(step, "DRTK_rast_db", rast_db)
        
        return rast, rast_db


def interpolate(
    attr: torch.Tensor,
    rast: torch.Tensor,
    faces: torch.Tensor,
    rast_db: Optional[torch.Tensor] = None,
    ctx: Optional['DRTKContext'] = None,
    peeler: Optional['DepthPeeler'] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Interpolate vertex attributes using DRTK.
    
    Args:
        attr: Vertex attributes [1, V, C] or [V, C]
        rast: Rasterization output from DRTKContext.rasterize()
        faces: Face indices [F, 3]
        rast_db: Unused, for compatibility with nvdiffrast API
        ctx: DRTKContext instance (uses cached rasterization outputs if provided)
        peeler: DepthPeeler instance (uses cached rasterization outputs if provided)
        
    Returns:
        interpolated: [1, H, W, C] interpolated attributes
        derivs: None (DRTK doesn't use derivatives this way)
    """
    import drtk
    
    if is_debug_enabled():
        step = next_step()
        dbg_tensor(step, "DRTK_interp_input_attr", attr, save=False)
        dbg_tensor(step, "DRTK_interp_input_rast", rast, save=False)
        dbg_value(step, "DRTK_interp_input_faces_shape", faces.shape)
    
    if attr.dim() == 2:
        attr = attr.unsqueeze(0)
    
    h, w = rast.shape[1], rast.shape[2]
    
    # Always reconstruct index_img from rast[...,3] since it correctly encodes triangle ID
    # (rast[...3] = index_img + 1 for geometry, 0 for background)
    # The peeler cache may have index_img filtered to -1 for depth peeling purposes,
    # but rast still has the original triangle IDs needed for attribute interpolation.
    index_img = (rast[..., 3] - 1).to(torch.int32)
    
    # Use cached bary_img if available (DRTK barycentrics are more accurate than reconstruction)
    if peeler is not None and peeler._cache_bary is not None:
        bary_img = peeler._cache_bary
    elif ctx is not None and ctx._cache_bary is not None and ctx._cache_resolution == (h, w):
        bary_img = ctx._cache_bary
    else:
        u = rast[..., 0]
        v = rast[..., 1]
        w0 = 1.0 - u - v
        bary_img = torch.stack([w0, u, v], dim=1 if rast.dim() == 4 else 0)
    
    # Ensure batch dimensions
    if index_img.dim() == 2:
        index_img = index_img.unsqueeze(0)
    if bary_img.dim() == 3:
        bary_img = bary_img.unsqueeze(0)
    
    if is_debug_enabled():
        step = next_step()
        dbg_tensor(step, "DRTK_interp_index_img", index_img, save=False)
        dbg_tensor(step, "DRTK_interp_bary_img", bary_img, save=False)
    
    faces_int = faces.to(torch.int32) if faces.dtype != torch.int32 else faces
    
    result = drtk.interpolate(attr, faces_int, index_img, bary_img)
    
    if is_debug_enabled():
        step = next_step()
        dbg_tensor(step, "DRTK_interp_raw_result", result, save=False)
    
    result = result.permute(0, 2, 3, 1)
    
    # Zero out background pixels (where index_img == -1)
    # This matches nvdiffrast's behavior
    bg_mask = (index_img < 0).unsqueeze(-1).expand_as(result)
    result = torch.where(bg_mask, torch.zeros_like(result), result)
    
    if is_debug_enabled():
        step = next_step()
        dbg_tensor(step, "DRTK_interp_final_result", result)
    
    return result, None


def texture(
    tex: torch.Tensor,
    uv: torch.Tensor,
    uv_da: Optional[torch.Tensor] = None,
    filter_mode: str = 'linear',
    boundary_mode: str = 'wrap',
) -> Tuple[torch.Tensor, None]:
    """Sample texture using DRTK's mipmap_grid_sample.
    
    Args:
        tex: Texture [1, C, H, W] or [C, H, W]
        uv: UV coordinates [1, H, W, 2]
        uv_da: UV derivatives (for mipmap level)
        filter_mode: 'linear', 'linear-mipmap-linear', 'nearest'
        boundary_mode: 'wrap', 'clamp', 'cube'
        
    Returns:
        sampled: [1, H, W, C] sampled texture
        None: placeholder for nvdiffrast API compatibility
    """
    import drtk
    
    # Ensure batch dimensions
    if tex.dim() == 3:
        tex = tex.unsqueeze(0)
    if uv.dim() == 3:
        uv = uv.unsqueeze(0)
    
    # Handle boundary mode
    padding_mode = 'border' if boundary_mode == 'clamp' else 'zeros'
    if boundary_mode == 'wrap':
        # DRTK doesn't have 'wrap' mode, need to handle with modulo
        uv = uv % 1.0
        padding_mode = 'border'  # After modulo, use border for edges
    
    if boundary_mode == 'cube':
        # Cubemap sampling not directly supported by DRTK
        # Fall back to PyTorch grid_sample (not differentiable w.r.t. pixels, but works)
        # Actually we should handle cubemap separately in the caller
        # For now, raise an error
        raise NotImplementedError("Cubemap sampling requires custom implementation. Use grid_sample with manual face selection.")
    
    # Determine if we need mipmap sampling
    use_mipmap = 'mipmap' in filter_mode or (uv_da is not None)
    
    if use_mipmap:
        # Build mipmap pyramid
        max_levels = int(torch.log2(torch.tensor(max(tex.shape[-2:]))).item()) + 1
        mipmap = build_mipmap(tex, max_levels)
        
        # Compute UV Jacobian for mipmap level selection
        if uv_da is not None:
            # uv_da is the derivative of UV w.r.t. pixels, shape [1, H, W, 2, 2] or [H, W, 2, 2]
            # This is already what we need for vt_dxdy_img
            vt_dxdy = uv_da if uv_da.dim() == 5 else uv_da.unsqueeze(0)
        else:
            # Compute Jacobian using finite differences
            vt_dxdy = compute_uv_jacobian(uv[0], tex.shape[-1])  # [H, W, 2, 2]
            vt_dxdy = vt_dxdy.unsqueeze(0)  # [1, H, W, 2, 2]
        
        # Use DRTK's mipmap grid sample
        # Note: mipmap_grid_sample expects UV in [-1, 1] range
        uv_grid = uv * 2 - 1  # [0, 1] -> [-1, 1]
        
        sampled = drtk.mipmap_grid_sample(
            mipmap,
            uv_grid,
            vt_dxdy,
            max_aniso=1,
            mode='bilinear' if 'linear' in filter_mode else 'nearest',
            padding_mode=padding_mode,
            align_corners=False,
        )
    else:
        # Simple bilinear/nearest sampling using grid_sample
        uv_grid = uv * 2 - 1  # [0, 1] -> [-1, 1]
        mode = 'bilinear' if filter_mode == 'linear' else 'nearest'
        sampled = F.grid_sample(tex, uv_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    
    # Convert from [N, C, H, W] to [N, H, W, C]
    result = sampled.permute(0, 2, 3, 1)
    
    return result, None


def antialias(color: torch.Tensor, rast: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Antialias using DRTK's edge_grad_estimator.
    
    Note: This is not semantic antialiasing like nvdiffrast. It provides differentiability
    at edge discontinuities. For visual antialiasing, post-processing may be needed.
    
    Args:
        color: Color image [1, H, W, C]
        rast: Rasterization output
        vertices: Vertices (unused, for API compatibility)
        faces: Faces (unused, for API compatibility)
        
    Returns:
        Color tensor with edge gradients attached
    """
    # DRTK's edge_grad_estimator is for backprop, not visual AA
    # For visual antialiasing, we could use multisampling or post-process AA
    # For now, return color unchanged (inference doesn't need AA for quality)
    # For training, would need edge_grad_estimator with proper setup
    return color


class DepthPeeler:
    """Context manager for depth peeling, mimicking nvdiffrast's DepthPeeler.
    
    DRTK doesn't have built-in depth peeling, so we implement it manually using
    a multi-pass approach with per-pixel depth comparison.
    
    Algorithm:
    - Layer 0: Normal rasterization (closest surface)
    - Layer N > 0: Re-rasterize, then filter results to exclude surfaces at or 
      in front of the minimum depth from previous layers.
      
    This approach has limitations compared to nvdiffrast's depth peeling:
    - It can only see surfaces that DRTK's rasterizer returns as "closest" at each pixel
    - For surfaces that were behind previous layers at EVERY pixel, they will now become
      visible at pixels where previous-layer surfaces were drawn
    - Complete depth peeling would require modifying vertex depths, which is expensive
    
    For transparent rendering in TRELLIS, this approximation is often sufficient because
    transparent surfaces tend to be spread across different screen-space regions.
    """
    def __init__(self, ctx: DRTKContext, vertices_clip: torch.Tensor, faces: torch.Tensor, resolution: Tuple[int, int]):
        self.ctx = ctx
        self.vertices_clip = vertices_clip
        self.faces = faces
        self.resolution = resolution
        self.layers_drawn = 0
        self.max_layers = 100
        self.min_depth_per_pixel = None  # [H, W] - closest depth seen so far
        self._cache_index_img = None
        self._cache_bary = None
    
    def _rasterize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform rasterization."""
        rast, rast_db = self.ctx.rasterize(self.vertices_clip, self.faces, self.resolution)
        return rast, rast_db
    
    def rasterize_next_layer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize the next depth layer (peel).
        
        Returns:
            rast: Rasterization output for current layer  
            rast_db: Barycentric derivatives (placeholder zeros)
        """
        import drtk
        
        h, w = self.resolution
        device = self.vertices_clip.device
        batch_size = self.vertices_clip.shape[0]
        
        if self.layers_drawn >= self.max_layers:
            return torch.zeros(batch_size, h, w, 4, device=device, dtype=torch.float32), None
        
        w_clip = self.vertices_clip[..., 3].clamp(min=1e-8)
        x_ndc = self.vertices_clip[..., 0] / w_clip
        y_ndc = -self.vertices_clip[..., 1] / w_clip
        z_ndc = self.vertices_clip[..., 2] / w_clip
        z_cam = self.vertices_clip[..., 3].clone()
        
        x_pix = (x_ndc + 1) * 0.5 * w - 0.5
        y_pix = (h - 1) - ((y_ndc + 1) * 0.5 * h - 0.5)
        
        v_pix = torch.stack([x_pix, y_pix, z_cam], dim=-1)
        faces_int = self.faces.to(torch.int32) if self.faces.dtype != torch.int32 else self.faces
        
        # Rasterize to get closest surfaces at each pixel
        index_img = drtk.rasterize(v_pix, faces_int, height=h, width=w)
        depth, bary = drtk.render(v_pix, faces_int, index_img)
        
        # Save ORIGINAL index_img BEFORE any filtering for use in rast[..., 3]
        # This is needed because drtk.rasterize always returns the closest surface,
        # but we need the original triangle IDs for attribute interpolation
        original_index_img = index_img.clone()
        
        # DRTK returns: depth [batch, H, W], bary [batch, 3, H, W]
        current_depth = depth[0]  # [H, W] camera-space depth
        
        if self.min_depth_per_pixel is None:
            # First layer: store depths, nothing to filter
            self.min_depth_per_pixel = current_depth.clone()
        else:
            # Subsequent layers: filter out surfaces at or in front of previous minimum
            # index_img [1, H, W], depth [1, H, W] -> current_depth [H, W]
            has_geom = index_img[0] >= 0  # [H, W]
            # Surface is valid for this layer if it's strictly behind previous minimum
            # (depth in camera space: larger = further from camera)
            behind_prev = current_depth > self.min_depth_per_pixel + 1e-6  # [H, W]
            
            # Valid pixel: has geometry AND is behind previous layers
            valid = has_geom & behind_prev  # [H, W]
            
            # Clear index_img where not valid
            index_img = torch.where(valid.unsqueeze(0), index_img, torch.full_like(index_img, -1))
            
            # Update min depth where valid
            self.min_depth_per_pixel = torch.where(
                valid,
                current_depth,
                self.min_depth_per_pixel
            )
        
        # Cache for interpolate
        self._cache_index_img = index_img  # This may be filtered for depth peeling
        self._cache_bary = bary
        
        # Build rast output in nvdiffrast format
        # DRTK returns: depth [batch, H, W], bary [batch, 3, H, W]
        # IMPORTANT: rast[..., 3] should contain the ORIGINAL triangle ID (not filtered)
        # so that interpolate() can reconstruct index_img from rast correctly
        rast = torch.zeros(batch_size, h, w, 4, device=device, dtype=torch.float32)
        rast[0, ..., 0] = bary[0, 1]  # u = weight for vertex 1
        rast[0, ..., 1] = bary[0, 2]  # v = weight for vertex 2
        rast[0, ..., 2] = depth[0]    # Camera-space depth for depth peeling
        # Use UNFILTERED (original) index_img for rast[..., 3]
        rast[0, ..., 3] = (original_index_img[0].float() + 1).float()
        
        rast_db = torch.zeros_like(rast)
        
        self.layers_drawn += 1
        
        return rast, rast_db
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# Module-level placeholder for compatibility
def RasterizeCudaContext(device: str = 'cuda'):
    """Create a DRTK context (stateless)."""
    return DRTKContext(device=device)