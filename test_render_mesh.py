import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

import os
os.environ['TRELLIS_DEBUG'] = '1'
os.environ['TRELLIS_DEBUG_DIR'] = 'test_output/debug'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import torch
from PIL import Image
import trimesh
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations.mesh import Mesh
from trellis2.utils import render_utils
from trellis2.utils.debug_utils import reset_debug_step, is_debug_enabled

IMAGE_PATH = "assets/example_image/T.png"
OUTPUT_DIR = "test_output"
SEED = 42
PIPELINE_TYPE = "512"
RENDER_RESOLUTION = 512
N_VIEWS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

reset_debug_step()

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open(IMAGE_PATH)
mesh_with_voxel = pipeline.run(image, seed=SEED, pipeline_type=PIPELINE_TYPE)[0]
mesh_with_voxel.simplify(16777216)

print(f"Generated mesh: {mesh_with_voxel.vertices.shape[0]} vertices, {mesh_with_voxel.faces.shape[0]} faces")

reset_debug_step()

plain_mesh = Mesh(mesh_with_voxel.vertices, mesh_with_voxel.faces)

extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws=[(-16 / 180 * np.pi)],
    pitchs=[20 / 180 * np.pi],
    rs=[10],
    fovs=[8],
)

renderer = render_utils.get_renderer(
    plain_mesh,
    resolution=RENDER_RESOLUTION,
    near=1,
    far=100,
    ssaa=2,
    chunk_size=None,
)

for i, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
    result = renderer.render(plain_mesh, extr, intr)

    normal = result.normal
    if normal.dim() == 2:
        normal = normal.unsqueeze(0).repeat(3, 1, 1)
    normal_np = np.clip(normal.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(normal_np).save(os.path.join(OUTPUT_DIR, f"normal_{i:02d}.png"))

    depth = result.depth
    if depth.dim() == 2:
        pass
    else:
        depth = depth.squeeze(0).squeeze(0) if depth.dim() == 3 else depth
    depth_np = np.clip(depth.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    Image.fromarray(depth_np).save(os.path.join(OUTPUT_DIR, f"depth_{i:02d}.png"))

    mask = result.mask
    if mask.dim() == 2:
        pass
    else:
        mask = mask.squeeze(0) if mask.dim() == 3 else mask
    mask_np = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(os.path.join(OUTPUT_DIR, f"mask_{i:02d}.png"))

    print(f"  View {i}: rendered")

simple_mesh = trimesh.Trimesh(
    vertices=mesh_with_voxel.vertices.detach().cpu().numpy(),
    faces=mesh_with_voxel.faces.detach().cpu().numpy(),
    process=False,
)
obj_path = os.path.join(OUTPUT_DIR, "mesh.obj")
simple_mesh.export(obj_path)
print(f"Saved OBJ: {obj_path}")

print(f"\nDone! View rendered, outputs in {OUTPUT_DIR}/")
print(f"Debug .npy files saved in {os.environ['TRELLIS_DEBUG_DIR']}/")
print(f"  - P*_*.npy: Pipeline stage tensors")
print(f"  - D*_*.npy: DRTK compat layer tensors")
print(f"  - R*_*.npy: MeshRenderer tensors")
print(f"  - NVR_R*_*.npy: nvdiffrast-specific comparison (this project uses DRTK)")