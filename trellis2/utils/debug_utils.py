import os
import numpy as np
import logging
import torch

_TRELLIS_DEBUG = os.environ.get('TRELLIS_DEBUG', '0') == '1'
_DEBUG_DIR = os.environ.get('TRELLIS_DEBUG_DIR', 'test_output/debug')
_DEBUG_STEP = [0]

def is_debug_enabled():
    return _TRELLIS_DEBUG

def get_debug_dir():
    return _DEBUG_DIR

def reset_debug_step():
    _DEBUG_STEP[0] = 0

def next_step():
    _DEBUG_STEP[0] += 1
    return _DEBUG_STEP[0]

def dbg_tensor(step, name, tensor, save=True, log_stats=True):
    if not _TRELLIS_DEBUG:
        return
    if tensor is None:
        if log_stats:
            logging.info(f"[DEBUG {step}] {name}: None")
        return
    try:
        t = tensor.detach().cpu().float()
    except Exception as e:
        if log_stats:
            logging.info(f"[DEBUG {step}] {name}: Failed to convert - {e}")
        return
    
    shape = list(t.shape)
    dtype = str(tensor.dtype)
    if t.numel() > 0:
        tmin = t.min().item()
        tmax = t.max().item()
        tmean = t.mean().item()
        tstd = t.std().item() if t.numel() > 1 else 0.0
        nonzero = int((t != 0).sum().item())
        total = t.numel()
    else:
        tmin = tmax = tmean = tstd = 0.0
        nonzero = total = 0
    
    if log_stats:
        logging.info(f"[DEBUG {step}] {name}: shape={shape} dtype={dtype} min={tmin:.6f} max={tmax:.6f} mean={tmean:.6f} std={tstd:.6f} nonzero={nonzero}/{total}")
    
    if save:
        os.makedirs(_DEBUG_DIR, exist_ok=True)
        safe_name = name.replace(' ', '_').replace('/', '_')
        np.save(os.path.join(_DEBUG_DIR, f"{step:02d}_{safe_name}.npy"), t.numpy())

def dbg_value(step, name, value, log_stats=True):
    if not _TRELLIS_DEBUG:
        return
    if log_stats:
        logging.info(f"[DEBUG {step}] {name}: {value}")

def dbg_rast_stats(step, rast, prefix="rast"):
    if not _TRELLIS_DEBUG:
        return
    if rast is None:
        logging.info(f"[DEBUG {step}] {prefix}: None")
        return
    
    t = rast.detach().cpu().float()
    mask = t[..., -1] > 0 if t.dim() == 4 else t[..., 3] > 0
    has_geom = mask.sum().item()
    total = mask.numel()
    
    if has_geom > 0:
        if t.dim() == 4:
            bary_u = t[0, ..., 0][mask[0]]
            bary_v = t[0, ..., 1][mask[0]]
            depth_vals = t[0, ..., 2][mask[0]]
            tri_ids = t[0, ..., 3][mask[0]]
        else:
            bary_u = t[..., 0][mask]
            bary_v = t[..., 1][mask]
            depth_vals = t[..., 2][mask]
            tri_ids = t[..., 3][mask]
        
        logging.info(f"[DEBUG {step}] {prefix}_coverage: {has_geom}/{total} pixels have geometry ({100*has_geom/total:.2f}%)")
        logging.info(f"[DEBUG {step}] {prefix}_bary_u: min={bary_u.min().item():.6f} max={bary_u.max().item():.6f} mean={bary_u.mean().item():.6f}")
        logging.info(f"[DEBUG {step}] {prefix}_bary_v: min={bary_v.min().item():.6f} max={bary_v.max().item():.6f} mean={bary_v.mean().item():.6f}")
        logging.info(f"[DEBUG {step}] {prefix}_depth: min={depth_vals.min().item():.6f} max={depth_vals.max().item():.6f} mean={depth_vals.mean().item():.6f}")
        logging.info(f"[DEBUG {step}] {prefix}_tri_ids: min={tri_ids.min().item():.1f} max={tri_ids.max().item():.1f} unique={len(torch.unique(tri_ids))}")
    else:
        logging.info(f"[DEBUG {step}] {prefix}: No geometry rendered (mask sum = 0)")