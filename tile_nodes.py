"""
Florence2 Tile Prompt Nodes

Integrates per-tile VLM captioning with tiled upscaling workflows.
Designed to work with ComfyUI_SimpleTiles_Uprez for advanced blending.

Author: Cedar
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import comfy.samplers
import comfy.sample
import folder_paths


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
                "florence_model": ("FLORENCE2",),
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
        prompts = []
        batch_size = tiles.shape[0]
        
        # Map task names to Florence2 prompts
        task_map = {
            "caption": "<CAPTION>",
            "detailed_caption": "<DETAILED_CAPTION>",
            "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        }
        
        processor = florence_model.get('processor')
        model = florence_model.get('model')
        
        if processor is None or model is None:
            raise ValueError("Invalid Florence2 model - missing processor or model")
        
        device = next(model.parameters()).device
        
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
                    row, col = pos[0], pos[1]
                    task_prompt = f"{task_prompt} This is tile at row {row}, column {col}."
            
            # Process through Florence2
            inputs = processor(
                text=task_prompt,
                images=tile_pil,
                return_tensors="pt"
            ).to(device)
            
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
        for token in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
            if text.startswith(token):
                text = text[len(token):].strip()
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
        
        batch_size = tiles_tensor.shape[0]
        processed = []
        
        # Get fallback conditioning
        fallback_cond = None
        if fallback_positive:
            fallback_cond = fallback_positive[0] if isinstance(fallback_positive, list) else fallback_positive
        
        for i in range(batch_size):
            # Get conditioning for this tile
            tile_cond = None
            if per_tile_positive and i < len(per_tile_positive):
                tile_cond = per_tile_positive[i]
            elif fallback_cond:
                tile_cond = fallback_cond
            else:
                raise ValueError(f"No conditioning available for tile {i}")
            
            # Extract single tile
            tile = tiles_tensor[i:i+1]
            
            # Encode to latent space
            tile_latent = vae.encode(tile[:, :, :, :3])  # Ensure 3 channels
            
            # Prepare latent dict
            latent = {"samples": tile_latent}
            
            # Sample with per-tile seed for variety
            tile_seed = seed_val + i
            
            # Run sampling
            samples = comfy.sample.sample(
                model=model,
                noise=comfy.sample.prepare_noise(latent["samples"], tile_seed),
                steps=steps_val,
                cfg=cfg_val,
                sampler_name=sampler,
                scheduler=sched,
                positive=tile_cond,
                negative=negative,
                latent_image=latent["samples"],
                denoise=denoise_val,
            )
            
            # Decode back to image
            decoded = vae.decode(samples)
            processed.append(decoded)
        
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


# Node registration
NODE_CLASS_MAPPINGS = {
    "Florence2BatchCaption": Florence2BatchCaption,
    "PromptListToConditioning": PromptListToConditioning,
    "TiledSamplerWithPromptList": TiledSamplerWithPromptList,
    "TilePromptPreview": TilePromptPreview,
    "PromptListEditor": PromptListEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2BatchCaption": "Florence2 Batch Caption (Tiles)",
    "PromptListToConditioning": "Prompt List → Conditioning",
    "TiledSamplerWithPromptList": "Tiled Sampler with Prompt List",
    "TilePromptPreview": "Tile Prompt Preview",
    "PromptListEditor": "Prompt List Editor",
}
