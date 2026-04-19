from pathlib import Path
from typing import Union
import trimesh


def fix_glb_buffer_view_targets(glb_path: Union[str, Path]) -> None:
    from pygltflib import GLTF2
    
    glb = GLTF2.load(str(glb_path))
    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963
    
    if not glb.meshes:
        return
    
    image_buffer_views = set()
    if glb.images:
        for img in glb.images:
            if img.bufferView is not None:
                image_buffer_views.add(img.bufferView)
    
    index_accessors = set()
    for mesh in glb.meshes:
        for prim in mesh.primitives:
            if prim.indices is not None:
                index_accessors.add(prim.indices)
    
    attribute_names = [
        "POSITION", "NORMAL", "TANGENT", "TEXCOORD_0", "TEXCOORD_1",
        "COLOR_0", "COLOR_1", "JOINTS_0", "JOINTS_1", "WEIGHTS_0", "WEIGHTS_1"
    ]
    attribute_accessors = set()
    for mesh in glb.meshes:
        for prim in mesh.primitives:
            if prim.attributes:
                for attr_name in attribute_names:
                    attr_idx = getattr(prim.attributes, attr_name, None)
                    if attr_idx is not None:
                        attribute_accessors.add(attr_idx)
    
    fixed_count = 0
    
    for acc_idx in index_accessors:
        if acc_idx < len(glb.accessors):
            acc = glb.accessors[acc_idx]
            if acc.bufferView is not None and acc.bufferView < len(glb.bufferViews):
                if acc.bufferView not in image_buffer_views:
                    bv = glb.bufferViews[acc.bufferView]
                    if bv.target is None:
                        bv.target = ELEMENT_ARRAY_BUFFER
                        fixed_count += 1
    
    for acc_idx in attribute_accessors:
        if acc_idx < len(glb.accessors):
            acc = glb.accessors[acc_idx]
            if acc.bufferView is not None and acc.bufferView < len(glb.bufferViews):
                if acc.bufferView not in image_buffer_views:
                    bv = glb.bufferViews[acc.bufferView]
                    if bv.target is None:
                        bv.target = ARRAY_BUFFER
                        fixed_count += 1
    
    if fixed_count > 0:
        glb.save(str(glb_path))


def validate_glb(glb_path: Union[str, Path]) -> bool:
    from pygltflib import GLTF2
    
    glb = GLTF2.load(str(glb_path))
    issues = []
    
    if not glb.scenes:
        issues.append("ERROR: No scenes defined")
    
    for mesh_idx, mesh in enumerate(glb.meshes):
        for prim_idx, prim in enumerate(mesh.primitives):
            if prim.attributes.POSITION is None:
                issues.append(f"ERROR: Mesh {mesh_idx} primitive {prim_idx} missing POSITION")
    
    image_buffer_views = set()
    if glb.images:
        for img in glb.images:
            if img.bufferView is not None:
                image_buffer_views.add(img.bufferView)
    
    for i, bv in enumerate(glb.bufferViews):
        if bv.target is None and i not in image_buffer_views:
            issues.append(f"WARNING: bufferView {i} missing target")
    
    has_errors = any("ERROR" in issue for issue in issues)
    return not has_errors


def export_glb_fixed(mesh: trimesh.Trimesh, path: Union[str, Path], **kwargs) -> None:
    mesh.export(str(path), **kwargs)
    fix_glb_buffer_view_targets(path)