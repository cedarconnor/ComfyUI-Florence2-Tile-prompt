"""
Florence2 Tile Prompt Nodes

Integrates per-tile VLM captioning with tiled upscaling workflows.
Designed to work with ComfyUI_SimpleTiles_Uprez for advanced blending.

Author: Cedar
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any, Union
import math
import comfy.samplers
import comfy.sample
import folder_paths
from comfy.utils import ProgressBar


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor (B,H,W,C) to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first in batch
    # Convert from float [0,1] to uint8 [0,255]
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor (1,H,W,C)."""
    np_image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    return tensor


class Florence2BatchCaption:
    """
    Takes a batch of tile images and generates a caption for each tile.
    Returns prompts as a list that can be used for per-tile conditioning.
    
    This enables intelligent upscaling where each tile receives its own
    context-aware prompt rather than a generic global prompt.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "florence_model": ("FL2MODEL",),
                "task": ([
                    "caption",
                    "detailed_caption", 
                    "more_detailed_caption",
                ], {"default": "detailed_caption"}),
                "prepend_text": ("STRING", {
                    "default": "high quality, detailed, ",
                    "multiline": False,
                }),
                "append_text": ("STRING", {
                    "default": ", sharp focus, professional",
                    "multiline": False,
                }),
                "max_tokens": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                }),
            },
            "optional": {
                "tile_calc": ("TILE_CALC",),
                "global_context": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompts_preview", "prompts_list",)
    OUTPUT_IS_LIST = (False, True,)
    
    FUNCTION = "batch_caption"
    CATEGORY = "Florence2/Tiles"

    def batch_caption(
        self,
        tiles: torch.Tensor,
        florence_model: Dict,
        task: str,
        prepend_text: str,
        append_text: str,
        max_tokens: int,
        tile_calc: Optional[Dict] = None,
        global_context: str = ""
    ) -> Tuple[str, List[str]]:
        """
        Process each tile through Florence2 and generate prompts.

        Args:
            tiles: Batched tensor of tiles (B, H, W, C)
            florence_model: Loaded Florence2 model dict
            task: Caption task type
            prepend_text: Text to prepend to each caption
            append_text: Text to append to each caption
            max_tokens: Maximum tokens for generation
            tile_calc: Optional tile calculation metadata
            global_context: Optional global context for better captions

        Returns:
            Tuple of (joined preview string, list of prompts)
        """
        # Input validation
        if tiles.dim() != 4:
            raise ValueError(f"Expected 4D tensor for tiles (B,H,W,C), got {tiles.dim()}D tensor with shape {tiles.shape}")

        batch_size = tiles.shape[0]
        if batch_size == 0:
            raise ValueError("Received empty batch of tiles")

        # Map task names to Florence2 prompts
        task_map = {
            "caption": "<CAPTION>",
            "detailed_caption": "<DETAILED_CAPTION>",
            "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        }

        processor = florence_model.get('processor')
        model = florence_model.get('model')
        model_dtype = florence_model.get('dtype')

        if processor is None or model is None:
            raise ValueError("Invalid Florence2 model - missing processor or model")

        device = next(model.parameters()).device

        # Validate tile_calc if provided
        if tile_calc and 'tile_positions' in tile_calc:
            if len(tile_calc['tile_positions']) < batch_size:
                print(f"Warning: tile_calc has {len(tile_calc['tile_positions'])} positions but got {batch_size} tiles")

        prompts = []
        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            # Extract single tile
            tile = tiles[i:i+1]
            tile_pil = tensor_to_pil(tile)
            
            # Build prompt with optional context
            task_prompt = task_map[task]
            if global_context:
                task_prompt = f"{task_prompt} Context: {global_context}"
            
            # Add tile position info if available
            if tile_calc and 'tile_positions' in tile_calc:
                pos = tile_calc['tile_positions'][i] if i < len(tile_calc['tile_positions']) else None
                if pos:
                    row = None
                    col = None
                    if isinstance(pos, dict):
                        row = pos.get("row")
                        col = pos.get("col")
                    elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        row, col = pos[0], pos[1]

                    if row is not None and col is not None:
                        task_prompt = f"{task_prompt} This is tile at row {row}, column {col}."
            
            # Process through Florence2
            inputs = processor(
                text=task_prompt,
                images=tile_pil,
                return_tensors="pt"
            )

            # Move inputs to correct device and dtype
            # Handle BatchEncoding properly - move each tensor individually
            inputs = {
                k: v.to(device=device, dtype=model_dtype if model_dtype and hasattr(v, 'dtype') and v.dtype.is_floating_point else v.dtype)
                if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=3,
                    early_stopping=True,
                )
            
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Parse output - Florence2 returns task token + result
            caption = self._parse_florence_output(generated_text, task_map[task])
            
            # Build final prompt
            prompt = f"{prepend_text}{caption}{append_text}".strip()
            prompts.append(prompt)

            # Update progress
            pbar.update(1)

        # Create preview string
        preview_parts = []
        for i, prompt in enumerate(prompts):
            preview_parts.append(f"[Tile {i}] {prompt}")
        preview = "\n\n".join(preview_parts)
        
        return (preview, prompts)
    
    def _parse_florence_output(self, text: str, task_token: str) -> str:
        """Extract the actual caption from Florence2 output."""
        # Florence2 outputs format: "<TASK_TOKEN>actual caption text"
        # Remove the task token if present
        text = text.strip()
        if text.startswith(task_token):
            text = text[len(task_token):].strip()
        return text


class PromptListToConditioning:
    """
    Converts a list of prompts into batched CONDITIONING for per-tile processing.
    Each prompt becomes conditioning for its corresponding tile index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
            },
            "optional": {
                "prompts_list": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning_list",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    
    FUNCTION = "encode_prompts"
    CATEGORY = "Florence2/Tiles"
    
    def encode_prompts(
        self,
        clip: List,
        prompts_list: Optional[List[str]] = None
    ) -> Tuple[List]:
        """
        Encode each prompt to CLIP embeddings, return as list.
        
        Args:
            clip: CLIP model (as list due to INPUT_IS_LIST)
            prompts_list: List of prompts to encode
            
        Returns:
            Tuple containing list of conditioning
        """
        # Handle list wrapping from INPUT_IS_LIST
        clip_model = clip[0] if isinstance(clip, list) else clip
        
        if prompts_list is None:
            prompts_list = [""]
        
        # Flatten if nested
        if isinstance(prompts_list, list) and len(prompts_list) == 1:
            if isinstance(prompts_list[0], list):
                prompts_list = prompts_list[0]
        
        conditioning_list = []
        
        for prompt in prompts_list:
            if not isinstance(prompt, str):
                prompt = str(prompt)
            
            # Encode with CLIP
            tokens = clip_model.tokenize(prompt)
            cond, pooled = clip_model.encode_from_tokens(
                tokens, 
                return_pooled=True
            )
            
            # ComfyUI conditioning format
            conditioning = [[cond, {"pooled_output": pooled}]]
            conditioning_list.append(conditioning)
        
        return (conditioning_list,)


class TiledSamplerWithPromptList:
    """
    KSampler that processes tiles with individual conditioning per tile.
    Essential for per-tile prompting workflow.
    
    Processes each tile independently with its own positive conditioning,
    enabling context-aware upscaling where different regions get 
    different prompts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tiles": ("IMAGE",),
                "vae": ("VAE",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "per_tile_positive": ("CONDITIONING",),
                "fallback_positive": ("CONDITIONING",),
                "tile_calc": ("TILE_CALC",),
            }
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_tiles",)
    
    FUNCTION = "sample_tiles"
    CATEGORY = "Florence2/Tiles"
    
    def sample_tiles(
        self,
        model: List,
        tiles: List[torch.Tensor],
        vae: List,
        negative: List,
        seed: List[int],
        steps: List[int],
        cfg: List[float],
        sampler_name: List[str],
        scheduler: List[str],
        denoise: List[float],
        per_tile_positive: Optional[List] = None,
        fallback_positive: Optional[List] = None,
        tile_calc: Optional[List] = None
    ) -> Tuple[torch.Tensor]:
        """
        Process each tile with optional per-tile conditioning.

        Args:
            model: Diffusion model
            tiles: Batched tile images
            vae: VAE for encoding/decoding
            negative: Negative conditioning (same for all tiles)
            seed: Base random seed
            steps: Sampling steps
            cfg: CFG scale
            sampler_name: Sampler algorithm
            scheduler: Noise scheduler
            denoise: Denoising strength
            per_tile_positive: Optional list of per-tile conditioning
            fallback_positive: Fallback conditioning if per_tile not available
            tile_calc: Optional tile metadata

        Returns:
            Processed tiles as batched tensor
        """
        # Unwrap list inputs
        model = model[0] if isinstance(model, list) else model
        tiles_tensor = tiles[0] if isinstance(tiles, list) else tiles
        vae = vae[0] if isinstance(vae, list) else vae
        negative = negative[0] if isinstance(negative, list) else negative
        seed_val = seed[0] if isinstance(seed, list) else seed
        steps_val = steps[0] if isinstance(steps, list) else steps
        cfg_val = cfg[0] if isinstance(cfg, list) else cfg
        sampler = sampler_name[0] if isinstance(sampler_name, list) else sampler_name
        sched = scheduler[0] if isinstance(scheduler, list) else scheduler
        denoise_val = denoise[0] if isinstance(denoise, list) else denoise

        # Input validation
        if tiles_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor for tiles (B,H,W,C), got {tiles_tensor.dim()}D with shape {tiles_tensor.shape}")

        batch_size = tiles_tensor.shape[0]
        if batch_size == 0:
            raise ValueError("Received empty batch of tiles")

        # Get fallback conditioning
        fallback_cond = None
        if fallback_positive:
            fallback_cond = fallback_positive[0] if isinstance(fallback_positive, list) else fallback_positive

        # Validate conditioning availability
        if per_tile_positive is None and fallback_cond is None:
            raise ValueError("Must provide either per_tile_positive or fallback_positive conditioning")

        if per_tile_positive and len(per_tile_positive) < batch_size and fallback_cond is None:
            raise ValueError(
                f"Insufficient conditioning: got {len(per_tile_positive)} conditioning for {batch_size} tiles "
                f"and no fallback_positive provided"
            )

        processed = []
        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            # Get conditioning for this tile
            tile_cond = None
            if per_tile_positive and i < len(per_tile_positive):
                tile_cond = per_tile_positive[i]
            elif fallback_cond:
                tile_cond = fallback_cond
            else:
                raise ValueError(f"No conditioning available for tile {i}/{batch_size}")

            # Extract single tile
            tile = tiles_tensor[i:i+1]

            # Handle channel count - ensure we have at least 3 channels
            num_channels = tile.shape[3]
            if num_channels < 3:
                raise ValueError(f"Tile {i} has {num_channels} channels, need at least 3 for RGB encoding")

            # Use only RGB channels (first 3) for VAE encoding
            tile_rgb = tile[:, :, :, :3]

            # Encode to latent space
            tile_latent = vae.encode(tile_rgb)

            # Extract samples from latent dict - VAE encode always returns dict in ComfyUI
            if not isinstance(tile_latent, dict) or "samples" not in tile_latent:
                raise RuntimeError(f"VAE encode returned unexpected format for tile {i}: {type(tile_latent)}")

            latent_samples = tile_latent["samples"]

            # Per-tile seed for variation
            tile_seed = seed_val + i

            # Prepare noise for this latent
            noise = comfy.sample.prepare_noise(latent_samples, tile_seed)

            # Sample
            sample_result = comfy.sample.sample(
                model=model,
                noise=noise,
                steps=steps_val,
                cfg=cfg_val,
                sampler_name=sampler,
                scheduler=sched,
                positive=tile_cond,
                negative=negative,
                latent_image=latent_samples,
                denoise=denoise_val,
            )

            # Decode - handle both dict and tensor returns
            if isinstance(sample_result, dict) and "samples" in sample_result:
                decoded = vae.decode(sample_result["samples"])
            else:
                decoded = vae.decode(sample_result)

            processed.append(decoded)

            # Update progress
            pbar.update(1)

        # Stack back to batch
        output = torch.cat(processed, dim=0)

        return (output,)


class TilePromptPreview:
    """
    Preview node that displays tiles alongside their generated prompts.
    Useful for debugging and verifying caption quality.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
            },
            "optional": {
                "prompts_preview": ("STRING", {"forceInput": True}),
                "tile_calc": ("TILE_CALC",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True
    
    FUNCTION = "preview"
    CATEGORY = "Florence2/Tiles"
    
    def preview(
        self,
        tiles: torch.Tensor,
        prompts_preview: str = "",
        tile_calc: Optional[Dict] = None
    ) -> Dict:
        """Generate preview information."""
        batch_size = tiles.shape[0]
        height, width = tiles.shape[1], tiles.shape[2]
        
        info_parts = [
            f"Tile Count: {batch_size}",
            f"Tile Size: {width}x{height}",
        ]
        
        if tile_calc:
            info_parts.extend([
                f"Grid Size: {tile_calc.get('grid_size', 'N/A')}",
                f"Overlap: {tile_calc.get('overlap_x', 'N/A')}x{tile_calc.get('overlap_y', 'N/A')}",
                f"Blend Mode: {tile_calc.get('blend_mode', 'linear')}",
            ])
        
        info_parts.append("\n--- Prompts ---")
        info_parts.append(prompts_preview if prompts_preview else "(No prompts)")
        
        info = "\n".join(info_parts)
        
        return {"ui": {"text": [info]}, "result": (info,)}


class PromptListEditor:
    """
    Edit individual prompts in a prompt list.
    Allows manual override of Florence2 captions for specific tiles.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255
                }),
                "new_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            },
            "optional": {
                "prompts_list": ("STRING", {"forceInput": True}),
            }
        }
    
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("edited_prompts",)
    
    FUNCTION = "edit_prompt"
    CATEGORY = "Florence2/Tiles"
    
    def edit_prompt(
        self,
        tile_index: List[int],
        new_prompt: List[str],
        prompts_list: Optional[List[str]] = None
    ) -> Tuple[List[str]]:
        """Replace a specific prompt in the list."""
        idx = tile_index[0] if isinstance(tile_index, list) else tile_index
        prompt = new_prompt[0] if isinstance(new_prompt, list) else new_prompt
        
        if prompts_list is None:
            return ([],)
        
        # Flatten nested lists
        if isinstance(prompts_list, list) and len(prompts_list) == 1:
            if isinstance(prompts_list[0], list):
                prompts_list = prompts_list[0]
        
        # Make a copy and edit
        edited = list(prompts_list)
        
        if 0 <= idx < len(edited) and prompt:
            edited[idx] = prompt
        
        return (edited,)


class TileCalcAddPositions:
    """
    Ensure tile_calc includes tile position metadata for downstream Florence2 usage.
    Adds tile_positions and tile_count when upstream split nodes omit them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_calc": ("TILE_CALC",),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                }),
                "overlap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("TILE_CALC",)
    RETURN_NAMES = ("tile_calc",)
    FUNCTION = "add_positions"
    CATEGORY = "Florence2/Tiles"

    def add_positions(
        self,
        tiles: torch.Tensor,
        tile_calc: Dict,
        tile_size: int = 0,
        overlap: int = 0,
    ) -> Tuple[Dict]:
        """
        Augment tile_calc with positions/tile_count if missing.

        Args:
            tiles: Batched tile images (B, H, W, C)
            tile_calc: Tile calculation metadata from DynamicTileSplit
            tile_size: Override tile size (0 = use from calc or infer)
            overlap: Override overlap (0 = use from calc)

        Returns:
            Updated tile_calc dictionary with tile_positions and tile_count
        """
        # Input validation
        if tiles.dim() != 4:
            raise ValueError(f"Expected 4D tensor for tiles (B,H,W,C), got {tiles.dim()}D with shape {tiles.shape}")

        updated = dict(tile_calc or {})

        # Derive tile size from actual tile dimensions
        tile_h = tiles.shape[1]
        tile_w = tiles.shape[2]

        # Check if tile_calc provides size info
        size_in_calc = updated.get("tile_size")
        if isinstance(size_in_calc, (list, tuple)) and len(size_in_calc) == 2:
            tile_w = int(size_in_calc[0])
            tile_h = int(size_in_calc[1])
        elif isinstance(size_in_calc, (int, float)) and size_in_calc > 0:
            tile_w = tile_h = int(size_in_calc)

        # Allow override via parameter
        if tile_size and tile_size > 0:
            tile_w = tile_h = int(tile_size)

        # Get overlap values
        overlap_x = int(updated.get("overlap_x", overlap))
        overlap_y = int(updated.get("overlap_y", overlap))
        overlap_x = max(overlap_x, 0)
        overlap_y = max(overlap_y, 0)

        stride_x = max(tile_w - overlap_x, 1)
        stride_y = max(tile_h - overlap_y, 1)

        # Determine grid layout
        count = tiles.shape[0]
        grid = updated.get("grid_size")

        if grid and isinstance(grid, (list, tuple)) and len(grid) == 2:
            # Use provided grid size
            rows, cols = int(grid[0]), int(grid[1])
        else:
            # Infer grid - try to make it close to square but prefer fewer rows
            # This is a guess since we don't know the original image dimensions
            if "original_size" in updated:
                orig_h, orig_w = updated["original_size"]
                # Calculate how many tiles would fit
                cols = max(1, int(math.ceil((orig_w - overlap_x) / stride_x)))
                rows = max(1, int(math.ceil((orig_h - overlap_y) / stride_y)))
                # Verify this matches tile count
                if rows * cols != count:
                    print(f"Warning: Calculated grid {rows}x{cols}={rows*cols} doesn't match tile count {count}")
                    # Fall back to square-ish layout
                    cols = int(math.ceil(math.sqrt(count)))
                    rows = int(math.ceil(count / cols))
            else:
                # No original size info - assume square-ish layout
                cols = int(math.ceil(math.sqrt(count)))
                rows = int(math.ceil(count / cols))

            updated["grid_size"] = (rows, cols)

        # Generate positions
        positions = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= count:
                    break
                x0 = c * stride_x
                y0 = r * stride_y
                positions.append((r, c, int(x0), int(y0), int(tile_w), int(tile_h)))
                idx += 1
            if idx >= count:
                break

        # Verify we generated the right number of positions
        if len(positions) != count:
            print(f"Warning: Generated {len(positions)} positions for {count} tiles")

        updated["tile_positions"] = positions
        updated["tile_count"] = count
        updated.setdefault("tile_size", (tile_w, tile_h))
        updated.setdefault("overlap_x", overlap_x)
        updated.setdefault("overlap_y", overlap_y)

        return (updated,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Florence2BatchCaption": Florence2BatchCaption,
    "PromptListToConditioning": PromptListToConditioning,
    "TiledSamplerWithPromptList": TiledSamplerWithPromptList,
    "TilePromptPreview": TilePromptPreview,
    "PromptListEditor": PromptListEditor,
    "TileCalcAddPositions": TileCalcAddPositions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2BatchCaption": "Florence2 Batch Caption (Tiles)",
    "PromptListToConditioning": "Prompt List -> Conditioning",
    "TiledSamplerWithPromptList": "Tiled Sampler with Prompt List",
    "TilePromptPreview": "Tile Prompt Preview",
    "PromptListEditor": "Prompt List Editor",
    "TileCalcAddPositions": "Tile Calc Add Positions",
}
