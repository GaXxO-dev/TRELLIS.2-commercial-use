from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel
from ..utils.debug_utils import is_debug_enabled, dbg_tensor, dbg_value, next_step


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        rembg_args = dict(args['rembg_model']['args'])
        rembg_args['model_name'] = 'ZhengPeng7/BiRefNet'
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**rembg_args)
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def _postprocess_decoded(
        self,
        decoded: torch.Tensor,
        fill_holes: bool,
        hole_structure: int,
        hole_iterations: int,
        hole_fill_algorithm: str,
        keep_only_shell: bool,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Post-process the decoded boolean voxel grid: optional hole filling
        and shell extraction.  Returns the (possibly modified) decoded tensor
        and a flag indicating whether scipy processing reordered axes.

        Returns:
            (decoded, axes_swapped) where axes_swapped is True when the
            hole-filling code reduced the tensor to 4D (B, D, H, W) requiring
            argwhere indexing [:, [0,1,2,3]] instead of the default [:, [0,2,3,4]].
        """
        axes_swapped = False
        if not fill_holes and not keep_only_shell:
            return decoded, axes_swapped

        try:
            from scipy.ndimage import binary_closing, label, binary_fill_holes, binary_erosion
        except ImportError:
            print("[Warning] scipy not installed, skipping hole filling / shell extraction.")
            return decoded, axes_swapped

        arr = decoded.cpu().numpy()
        if arr.ndim == 5:
            arr = arr[:, 0]
        closed = np.zeros_like(arr)

        for b in range(arr.shape[0]):
            filled = arr[b].astype(np.bool_)

            if fill_holes:
                inv = ~filled
                labeled, num_features = label(inv)
                border_mask = np.zeros_like(inv)
                border_mask[0, :, :] = border_mask[-1, :, :] = 1
                border_mask[:, 0, :] = border_mask[:, -1, :] = 1
                border_mask[:, :, 0] = border_mask[:, :, -1] = 1
                border_labels = np.unique(labeled[border_mask == 1])
                holes = np.isin(labeled, border_labels, invert=True) & (labeled > 0)
                n_holes = np.unique(labeled[holes]).size
                if verbose:
                    print(f"[Sparse HoleFill] Batch {b}: Found {n_holes} holes before filling.")

                if hole_fill_algorithm == "morphological_closing":
                    closed[b] = binary_closing(arr[b], structure=np.ones((hole_structure,) * 3), iterations=hole_iterations)
                elif hole_fill_algorithm == "flood_fill":
                    closed1 = binary_closing(arr[b], structure=np.ones((hole_structure,) * 3), iterations=hole_iterations)
                    filled_h = binary_fill_holes(closed1)
                    labeled2, num = label(filled_h)
                    if num > 0:
                        sizes = np.bincount(labeled2.ravel())
                        sizes[0] = 0
                        largest = sizes.argmax()
                        closed[b] = (labeled2 == largest)
                    else:
                        closed[b] = filled_h
                elif hole_fill_algorithm == "remove_small_holes":
                    try:
                        from skimage.morphology import remove_small_holes
                        temp = np.copy(arr[b])
                        for z in range(temp.shape[0]):
                            temp[z] = remove_small_holes(temp[z].astype(bool), area_threshold=hole_structure ** 2)
                        closed[b] = temp
                    except ImportError:
                        if verbose:
                            print("[Warning] scikit-image not installed, falling back to flood_fill.")
                        closed1 = binary_closing(arr[b], structure=np.ones((hole_structure,) * 3), iterations=hole_iterations)
                        filled_h = binary_fill_holes(closed1)
                        labeled2, num = label(filled_h)
                        if num > 0:
                            sizes = np.bincount(labeled2.ravel())
                            sizes[0] = 0
                            largest = sizes.argmax()
                            closed[b] = (labeled2 == largest)
                        else:
                            closed[b] = filled_h
                else:
                    if verbose:
                        print(f"[Sparse HoleFill] Unknown algorithm: {hole_fill_algorithm}, skipping.")
                    closed[b] = arr[b]

                if verbose:
                    filled2 = closed[b].astype(np.bool_)
                    inv2 = ~filled2
                    labeled2, num_features2 = label(inv2)
                    border_labels2 = np.unique(labeled2[border_mask == 1])
                    holes2 = np.isin(labeled2, border_labels2, invert=True) & (labeled2 > 0)
                    n_holes2 = np.unique(labeled2[holes2]).size
                    print(f"[Sparse HoleFill] Batch {b}: {n_holes - n_holes2} holes filled, {n_holes2} remain.")
            else:
                closed[b] = filled

            if keep_only_shell:
                filled_s = closed[b].astype(np.bool_)
                before_count = int(filled_s.sum())
                struct = np.ones((3, 3, 3), dtype=bool)
                eroded = binary_erosion(filled_s, structure=struct, border_value=0)
                eroded = binary_erosion(eroded, structure=struct, border_value=0)
                shell = filled_s & ~eroded
                closed[b] = shell
                after_count = int(shell.sum())
                if verbose:
                    print(f"[Sparse Shell] Batch {b}: {before_count} -> {after_count} voxels (removed {before_count - after_count} deeply interior)")

        decoded = torch.from_numpy(closed).to(decoded.device).contiguous().cpu()
        axes_swapped = True
        return decoded, axes_swapped

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        fill_holes: bool = False,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            fill_holes (bool): Whether to fill holes in the decoded voxel grid.
            hole_structure (int): Structure size for hole filling.
            hole_iterations (int): Iterations for hole filling.
            hole_fill_algorithm (str): Algorithm: 'morphological_closing', 'flood_fill', 'remove_small_holes'.
            keep_only_shell (bool): Whether to keep only the surface shell.
            dino_lock (float): DINO-lock base strength (0 = disabled).
            dino_substeps (int): Substeps during DINO foundation phase.
            dino_foundation_cap (float): DINO foundation cap strength.
            verbose (bool): Verbose output.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P1_ss_noise", noise)
            dbg_value(next_step(), "P1_ss_resolution", f"{resolution}, model_reso={reso}, in_channels={in_channels}")
        
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P2_ss_z_s_sampled", z_s)
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P3_ss_decoded", decoded.float())
        
        if resolution != decoded.shape[2]:
            if resolution < decoded.shape[2]:
                ratio = decoded.shape[2] // resolution
                decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            else:
                decoded = torch.nn.functional.interpolate(decoded.float(), size=(resolution, resolution, resolution), mode='nearest') > 0.5

        # Optional hole-filling + shell extraction
        decoded, axes_swapped = self._postprocess_decoded(
            decoded, fill_holes, hole_structure, hole_iterations,
            hole_fill_algorithm, keep_only_shell, verbose,
        )

        if axes_swapped:
            coords = torch.argwhere(decoded)[:, [0, 1, 2, 3]].int()
        else:
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        if is_debug_enabled():
            dbg_tensor(next_step(), "P4_ss_coords", coords)
            dbg_value(next_step(), "P4_ss_num_voxels", coords.shape[0])

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        verbose: bool = False,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            dino_lock (float): DINO-lock base strength (0 = disabled).
            dino_substeps (int): Substeps during DINO foundation phase.
            dino_foundation_cap (float): DINO foundation cap strength.
            verbose (bool): Verbose output.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P5_shape_slat_noise_feats", noise.feats)
            dbg_tensor(next_step(), "P5_shape_slat_noise_coords", noise.coords)
        
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P6_shape_slat_sampled", slat.feats)

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P7_shape_slat_denormed", slat.feats)
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        verbose: bool = False,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            sparse_structure_resolution (int): Resolution of the sparse structure (affects token quantization).
            dino_lock (float): DINO-lock base strength (0 = disabled).
            dino_substeps (int): Substeps during DINO foundation phase.
            dino_foundation_cap (float): DINO foundation cap strength.
            verbose (bool): Verbose output.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        slat = slat.to(self.device)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution

        ratio = sparse_structure_resolution / 32

        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        verbose: bool = False,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
            dino_lock (float): DINO-lock base strength (0 = disabled).
            dino_substeps (int): Substeps during DINO foundation phase.
            dino_foundation_cap (float): DINO foundation cap strength.
            verbose (bool): Verbose output.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples
        if self.low_vram:
            flow_model.cpu()

        if is_debug_enabled():
            dbg_tensor(next_step(), "P6_texture_slat_sampled", slat.feats)

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P7_texture_slat_denormed", slat.feats)
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        if is_debug_enabled():
            dbg_tensor(next_step(), "P8_decode_shape_slat_input", shape_slat.feats)
        
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P9_decode_mesh_vertices", meshes[0].vertices)
            dbg_tensor(next_step(), "P9_decode_mesh_faces", meshes[0].faces)
            dbg_value(next_step(), "P9_mesh_stats", f"vertices={meshes[0].vertices.shape[0]} faces={meshes[0].faces.shape[0]}")
        
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        
        if is_debug_enabled():
            dbg_tensor(next_step(), "P10_final_mesh_vertices", out_mesh[0].vertices)
            dbg_tensor(next_step(), "P10_final_mesh_faces", out_mesh[0].faces)
            dbg_value(next_step(), "P10_final_mesh_stats", f"vertices={out_mesh[0].vertices.shape[0]} faces={out_mesh[0].faces.shape[0]}")
        
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        fill_holes: bool = False,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = False,
        verbose: bool = False,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
            sparse_structure_resolution (int): Resolution of the sparse structure.
            dino_lock (float): DINO-lock base strength (0 = disabled).
            dino_substeps (int): Substeps during DINO foundation phase.
            dino_foundation_cap (float): DINO foundation cap strength.
            fill_holes (bool): Whether to fill holes in the decoded voxel grid.
            hole_structure (int): Structure size for hole filling.
            hole_iterations (int): Iterations for hole filling.
            hole_fill_algorithm (str): Algorithm: 'morphological_closing', 'flood_fill', 'remove_small_holes'.
            keep_only_shell (bool): Whether to keep only the surface shell.
            verbose (bool): Verbose output.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        ss_res = sparse_structure_resolution
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params,
            fill_holes=fill_holes,
            hole_structure=hole_structure,
            hole_iterations=hole_iterations,
            hole_fill_algorithm=hole_fill_algorithm,
            keep_only_shell=keep_only_shell,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            verbose=verbose,
        )
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                dino_lock=dino_lock, dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap, verbose=verbose,
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params,
                dino_lock=dino_lock, dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap, verbose=verbose,
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                dino_lock=dino_lock, dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap, verbose=verbose,
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
                dino_lock=dino_lock, dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap, verbose=verbose,
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution=sparse_structure_resolution,
                dino_lock=dino_lock, dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap, verbose=verbose,
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
                dino_lock=dino_lock, dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap, verbose=verbose,
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution=sparse_structure_resolution,
                dino_lock=dino_lock, dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap, verbose=verbose,
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
                dino_lock=dino_lock, dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap, verbose=verbose,
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

    # =========================================================================
    # Multi-view methods
    # =========================================================================

    @torch.no_grad()
    def sample_sparse_structure_multiview(
        self,
        conds: dict,
        views: list,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        fill_holes: bool = False,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Sample sparse structures with multi-view blending.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)

        sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )

        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        if self.low_vram:
            flow_model.to(self.device)

        z_s = sampler.sample(
            flow_model,
            noise,
            conds=conds,
            **sampler_params,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            verbose=True,
            tqdm_desc="Sampling sparse structure (MultiView)",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples

        if self.low_vram:
            flow_model.cpu()

        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)

        decoded = decoder(z_s) > 0

        if self.low_vram:
            decoder.cpu()

        if resolution != decoded.shape[2]:
            if resolution < decoded.shape[2]:
                ratio = decoded.shape[2] // resolution
                decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            else:
                decoded = torch.nn.functional.interpolate(decoded.float(), size=(resolution, resolution, resolution), mode='nearest') > 0.5

        # Optional hole-filling + shell extraction
        decoded, axes_swapped = self._postprocess_decoded(
            decoded, fill_holes, hole_structure, hole_iterations,
            hole_fill_algorithm, keep_only_shell, verbose,
        )

        if axes_swapped:
            coords = torch.argwhere(decoded)[:, [0, 1, 2, 3]].int()
        else:
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        coords = coords.cpu()

        del decoded
        del z_s

        return coords

    @torch.no_grad()
    def sample_shape_slat_multiview(
        self,
        conds: dict,
        views: list,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        verbose: bool = False,
    ) -> SparseTensor:
        """Sample shape structured latent with multi-view blending."""
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )

        sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )

        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}

        if self.low_vram:
            flow_model.to(self.device)

        slat = sampler.sample(
            flow_model,
            noise,
            conds=conds,
            **sampler_params,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView)",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples

        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    @torch.no_grad()
    def sample_shape_slat_cascade_multiview(
        self,
        lr_conds: dict,
        conds: dict,
        views: list,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sparse_structure_resolution: int = 32,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        verbose: bool = False,
    ) -> SparseTensor:
        """Sample shape structured latent with multi-view blending (cascade)."""
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels, device=self.device),
            coords=coords,
        )

        sampler_lr = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model_lr.resolution,
        )

        sampler_params_combined = {**self.shape_slat_sampler_params, **sampler_params}

        if self.low_vram:
            flow_model_lr.to(self.device)

        slat = sampler_lr.sample(
            flow_model_lr,
            noise,
            conds=lr_conds,
            **sampler_params_combined,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView LR)",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples

        if self.low_vram:
            flow_model_lr.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        # Upsample
        slat = slat.to(self.device)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        ratio = sparse_structure_resolution / 32

        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                hr_resolution = 512
                break

        # HR
        sampler_hr = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )

        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )

        if self.low_vram:
            flow_model.to(self.device)

        d_slat = sampler_hr.sample(
            flow_model,
            noise,
            conds=conds,
            **sampler_params_combined,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView HR)",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples

        if self.low_vram:
            flow_model.cpu()

        slat = d_slat * std + mean

        return slat, hr_resolution

    @torch.no_grad()
    def sample_tex_slat_multiview(
        self,
        conds: dict,
        views: list,
        shape_slat: SparseTensor,
        flow_model,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        verbose: bool = False,
    ) -> SparseTensor:
        """Sample texture structured latent with multi-view blending."""
        # Normalize shape slat for conditioning
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_normalized = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))

        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}

        sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )

        if self.low_vram:
            flow_model.to(self.device)

        slat = sampler.sample(
            flow_model,
            noise,
            conds=conds,
            **sampler_params,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            concat_cond=shape_slat_normalized,
            verbose=True,
            tqdm_desc="Sampling texture SLat (MultiView)",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
        ).samples

        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    @torch.no_grad()
    def run_multiview(
        self,
        front: Image.Image,
        back: Image.Image = None,
        left: Image.Image = None,
        right: Image.Image = None,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        pipeline_type: Optional[str] = None,
        generate_texture_slat: bool = True,
        return_latent: bool = False,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        fill_holes: bool = False,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = False,
        verbose: bool = False,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline with named multi-view images and spatial blending.

        Args:
            front (Image.Image): Front view image (required).
            back (Image.Image): Back view image (optional).
            left (Image.Image): Left view image (optional).
            right (Image.Image): Right view image (optional).
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Parameters for the texture SLat sampler.
            max_num_tokens (int): The maximum number of tokens to use.
            sparse_structure_resolution (int): Resolution of the sparse structure.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            generate_texture_slat (bool): Whether to generate texture SLat.
            return_latent (bool): Whether to return the latent codes.
            front_axis (str): Front axis: 'z' or 'x'.
            blend_temperature (float): Blend temperature for view weight softmax.
            dino_lock (float): DINO-lock base strength (0 = disabled).
            dino_substeps (int): Substeps during DINO foundation phase.
            dino_foundation_cap (float): DINO foundation cap strength.
            fill_holes (bool): Whether to fill holes in the decoded voxel grid.
            hole_structure (int): Structure size for hole filling.
            hole_iterations (int): Iterations for hole filling.
            hole_fill_algorithm (str): Algorithm: 'morphological_closing', 'flood_fill', 'remove_small_holes'.
            keep_only_shell (bool): Whether to keep only the surface shell.
            verbose (bool): Verbose output.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        # Collect views
        views_dict = {'front': front}
        if back is not None:
            views_dict['back'] = back
        if left is not None:
            views_dict['left'] = left
        if right is not None:
            views_dict['right'] = right

        views_list = list(views_dict.keys())

        torch.manual_seed(seed)

        # 1. Conditioning: calculate per-view conditioning
        conds = {}        # 1024 or None (if 512)
        lr_conds = {}     # 512 (for cascade)
        conds_512 = {}    # Explicit 512 storage for structure sampling
        conds_1024 = {}

        if pipeline_type == '512':
            for v, img in views_dict.items():
                c = self.get_cond([img], 512)
                conds[v] = c
                conds_512[v] = c
        elif pipeline_type == '1024':
            for v, img in views_dict.items():
                c1024 = self.get_cond([img], 1024)
                conds[v] = c1024
                conds_1024[v] = c1024
                conds_512[v] = self.get_cond([img], 512)
        elif 'cascade' in pipeline_type:
            for v, img in views_dict.items():
                c512 = self.get_cond([img], 512)
                c1024 = self.get_cond([img], 1024)
                lr_conds[v] = c512
                conds[v] = c1024
                conds_512[v] = c512
                conds_1024[v] = c1024

        # 2. Sparse Structure MultiView
        coords = self.sample_sparse_structure_multiview(
            conds_512, views_list, sparse_structure_resolution,
            sampler_params=sparse_structure_sampler_params,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            fill_holes=fill_holes,
            hole_structure=hole_structure,
            hole_iterations=hole_iterations,
            hole_fill_algorithm=hole_fill_algorithm,
            keep_only_shell=keep_only_shell,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            verbose=verbose,
        )

        # 3. Shape Slat MultiView
        shape_slat = None
        res = 0

        if pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade_multiview(
                lr_conds, conds, views_list,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sparse_structure_resolution=sparse_structure_resolution,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap,
                verbose=verbose,
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade_multiview(
                lr_conds, conds, views_list,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sparse_structure_resolution=sparse_structure_resolution,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap,
                verbose=verbose,
            )
        elif pipeline_type == '512':
            shape_slat = self.sample_shape_slat_multiview(
                conds, views_list,
                self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap,
                verbose=verbose,
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat_multiview(
                conds, views_list,
                self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap,
                verbose=verbose,
            )
            res = 1024

        # 4. Texture Slat MultiView
        tex_slat = None
        if generate_texture_slat:
            if pipeline_type == '512':
                flow_model = self.models['tex_slat_flow_model_512']
                tex_conds = conds_512
            else:
                flow_model = self.models['tex_slat_flow_model_1024']
                tex_conds = conds_1024

            tex_slat = self.sample_tex_slat_multiview(
                tex_conds, views_list,
                shape_slat=shape_slat,
                flow_model=flow_model,
                sampler_params=tex_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap,
                verbose=verbose,
            )

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
