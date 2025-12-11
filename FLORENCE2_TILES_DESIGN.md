# Florence2 Tile Prompt + SimpleTiles Integration

## Overview

This design integrates **per-tile VLM captioning** using Florence2 with the **advanced blending** capabilities of SimpleTiles_Uprez. The goal is to enable intelligent upscaling where each tile receives its own context-aware prompt, then gets blended back seamlessly.

## Use Case

Traditional tiled upscaling uses a single global prompt for the entire image. This causes problems:
- A landscape image might have sky, mountains, and forest—each needing different enhancement
- Architectural images have different textures (brick, glass, foliage) that benefit from specific prompts
- Faces in different regions need different attention than background elements

**Solution**: Caption each tile independently with Florence2, then process each tile with its unique prompt before blending.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐     ┌───────────────────┐     ┌──────────────────────┐   │
│  │  INPUT   │────▶│  DynamicTileSplit │────▶│  tile_batch (IMAGE)  │   │
│  │  IMAGE   │     │  + tile_calc obj  │     │  + tile_calc obj     │   │
│  └──────────┘     └───────────────────┘     └──────────┬───────────┘   │
│                                                         │               │
│                                                         ▼               │
│                                           ┌──────────────────────────┐  │
│                                           │  Florence2BatchCaption   │  │
│                                           │  (per-tile captioning)   │  │
│                                           └──────────┬───────────────┘  │
│                                                      │                  │
│                                                      ▼                  │
│                                           ┌──────────────────────────┐  │
│                                           │  STRING[] prompts        │  │
│                                           │  (one per tile)          │  │
│                                           └──────────┬───────────────┘  │
│                                                      │                  │
│                         ┌────────────────────────────┤                  │
│                         │                            │                  │
│                         ▼                            ▼                  │
│            ┌────────────────────┐      ┌────────────────────────────┐  │
│            │  TiledKSampler /   │      │  Optional: Prompt          │  │
│            │  img2img process   │◀─────│  Conditioning per tile     │  │
│            │  with per-tile     │      └────────────────────────────┘  │
│            │  conditioning      │                                      │
│            └─────────┬──────────┘                                      │
│                      │                                                  │
│                      ▼                                                  │
│            ┌────────────────────┐                                      │
│            │  DynamicTileMerge  │                                      │
│            │  (advanced blend)  │                                      │
│            └─────────┬──────────┘                                      │
│                      │                                                  │
│                      ▼                                                  │
│            ┌────────────────────┐                                      │
│            │   OUTPUT IMAGE     │                                      │
│            └────────────────────┘                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## New Nodes to Implement

### 1. Florence2BatchCaption

Processes a batch of tile images through Florence2 and returns an array of prompts.

```python
class Florence2BatchCaption:
    """
    Takes a batch of tile images and generates a caption for each tile.
    Returns prompts as a list that can be used for per-tile conditioning.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),  # Batched tile images from DynamicTileSplit
                "florence_model": ("FLORENCE2",),  # Loaded Florence2 model
                "task": ([
                    "caption",           # Short caption
                    "detailed_caption",  # Longer description
                    "more_detailed_caption",  # Most detailed
                ], {"default": "detailed_caption"}),
                "prepend_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "high quality, detailed, "
                }),
                "append_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": ", 8k, sharp focus"
                }),
            },
            "optional": {
                "tile_calc": ("TILE_CALC",),  # For metadata/position awareness
                "global_context": ("STRING", {  # Optional global context
                    "default": "",
                    "multiline": True,
                    "placeholder": "This is a landscape photograph..."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompts", "prompts_list",)
    OUTPUT_IS_LIST = (False, True,)  # First is joined, second is list
    
    FUNCTION = "batch_caption"
    CATEGORY = "Florence2/Tiles"

    def batch_caption(self, tiles, florence_model, task, prepend_text, append_text,
                      tile_calc=None, global_context=""):
        """
        Process each tile through Florence2 and generate prompts.
        """
        prompts = []
        batch_size = tiles.shape[0]
        
        # Map task to Florence2 prompt
        task_map = {
            "caption": "<CAPTION>",
            "detailed_caption": "<DETAILED_CAPTION>",
            "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        }
        
        for i in range(batch_size):
            # Extract single tile (H, W, C) -> (1, H, W, C)
            tile = tiles[i:i+1]
            
            # Convert to PIL for Florence2
            tile_pil = tensor_to_pil(tile)
            
            # Run Florence2 inference
            caption = self.run_florence(
                florence_model, 
                tile_pil, 
                task_map[task],
                global_context
            )
            
            # Build final prompt
            prompt = f"{prepend_text}{caption}{append_text}"
            prompts.append(prompt.strip())
        
        # Return both joined string (for preview) and list (for per-tile use)
        joined_prompts = "\n---\n".join(prompts)
        
        return (joined_prompts, prompts)
    
    def run_florence(self, model, image, task_prompt, context=""):
        """Run Florence2 model on a single image."""
        # Implementation depends on kijai's Florence2 loader
        # This is the core inference call
        processor = model['processor']
        model_instance = model['model']
        
        prompt = task_prompt
        if context:
            prompt = f"{task_prompt} Context: {context}"
        
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model_instance.device)
        
        with torch.no_grad():
            generated_ids = model_instance.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
            )
        
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Parse the output (Florence2 returns task token + result)
        result = self.parse_florence_output(generated_text, task_prompt)
        return result
```

### 2. PromptListToConditioning

Converts a list of prompts into per-tile CONDITIONING that can be used with batch processing.

```python
class PromptListToConditioning:
    """
    Takes a list of prompts and a CLIP model, returns batched conditioning.
    Each prompt becomes conditioning for its corresponding tile index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": True}),  # List input
                "clip": ("CLIP",),
            },
            "optional": {
                "base_conditioning": ("CONDITIONING",),  # Optional to combine
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning_batch",)
    INPUT_IS_LIST = True
    
    FUNCTION = "encode_prompts"
    CATEGORY = "Florence2/Tiles"
    
    def encode_prompts(self, prompts, clip, base_conditioning=None):
        """
        Encode each prompt to CLIP embeddings, return as batch.
        """
        # Handle list input
        if isinstance(prompts, list) and len(prompts) == 1:
            # Could be a joined string, split it
            if isinstance(prompts[0], str) and "\n---\n" in prompts[0]:
                prompts = prompts[0].split("\n---\n")
            else:
                prompts = prompts
        
        conditioning_batch = []
        
        for prompt in prompts:
            # Encode single prompt
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            conditioning_batch.append([[cond, {"pooled_output": pooled}]])
        
        # Stack into batch format ComfyUI expects
        return (conditioning_batch,)
```

### 3. TiledKSamplerWithPromptList

A modified KSampler that accepts per-tile conditioning.

```python
class TiledKSamplerWithPromptList:
    """
    KSampler that processes tiles with individual conditioning per tile.
    Essential for per-tile prompting workflow.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tiles": ("IMAGE",),
                "tile_calc": ("TILE_CALC",),
                "positive": ("CONDITIONING",),  # Can be list or single
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "per_tile_positive": ("CONDITIONING",),  # List of conditioning
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_tiles",)
    
    FUNCTION = "sample_tiles"
    CATEGORY = "Florence2/Tiles"
    
    def sample_tiles(self, model, tiles, tile_calc, positive, negative, vae,
                     seed, steps, cfg, sampler_name, scheduler, denoise,
                     per_tile_positive=None):
        """
        Process each tile with optional per-tile conditioning.
        """
        batch_size = tiles.shape[0]
        processed = []
        
        for i in range(batch_size):
            # Get conditioning for this tile
            if per_tile_positive is not None and i < len(per_tile_positive):
                tile_positive = per_tile_positive[i]
            else:
                tile_positive = positive
            
            # Extract single tile
            tile = tiles[i:i+1]
            
            # Encode to latent
            latent = vae.encode(tile)
            
            # Sample
            samples = common_ksampler(
                model, seed + i, steps, cfg, sampler_name, scheduler,
                tile_positive, negative, latent, denoise=denoise
            )
            
            # Decode
            decoded = vae.decode(samples)
            processed.append(decoded)
        
        # Stack back to batch
        return (torch.cat(processed, dim=0),)
```

---

## Integration with Existing SimpleTiles

### Modified DynamicTileSplit

Add metadata to `tile_calc` for downstream Florence2 processing:

```python
# In dynamic.py - DynamicTileSplit class

def split(self, image, tile_size, overlap, blend_mode="linear"):
    # ... existing split logic ...
    
    tile_calc = {
        'overlap_x': overlap_x,
        'overlap_y': overlap_y,
        'original_size': (height, width),
        'tile_size': tile_size,
        'grid_size': (rows, cols),
        'blend_mode': blend_mode,
        # NEW: Add position metadata for each tile
        'tile_positions': tile_positions,  # List of (row, col, x, y, w, h)
        'tile_count': len(tiles),
    }
    
    return (tiles_batch, tile_calc)
```

### Modified DynamicTileMerge

No changes needed—it already handles batched tiles and `tile_calc`.

---

## Complete Workflow Example

```python
# Pseudo-workflow showing node connections

# 1. Load image
image = LoadImage("input.png")

# 2. Split into tiles
tiles, tile_calc = DynamicTileSplit(
    image=image,
    tile_size=768,
    overlap=128,
    blend_mode="noise"  # or "laplacian"
)

# 3. Load Florence2 model
florence_model = DownloadAndLoadFlorence2Model(
    repo="microsoft/Florence-2-large"
)

# 4. Generate per-tile captions
prompts_joined, prompts_list = Florence2BatchCaption(
    tiles=tiles,
    florence_model=florence_model,
    task="detailed_caption",
    prepend_text="high quality, detailed, sharp, ",
    append_text=", professional photography, 8k"
)

# 5. Load SDXL for upscaling
model, clip, vae = CheckpointLoaderSimple("sdxl_base.safetensors")

# 6. Encode per-tile prompts to conditioning
per_tile_cond = PromptListToConditioning(
    prompts=prompts_list,
    clip=clip
)

# 7. Encode negative prompt (same for all tiles)
negative_cond = CLIPTextEncode(
    text="blurry, low quality, artifacts, noise",
    clip=clip
)

# 8. Process tiles with per-tile conditioning
processed_tiles = TiledKSamplerWithPromptList(
    model=model,
    tiles=tiles,
    tile_calc=tile_calc,
    per_tile_positive=per_tile_cond,
    negative=negative_cond,
    vae=vae,
    denoise=0.35,
    steps=20
)

# 9. Merge tiles with advanced blending
output = DynamicTileMerge(
    tiles=processed_tiles,
    tile_calc=tile_calc,
    blend=64  # Blend distance in pixels
)

# 10. Save
SaveImage(output, "upscaled_output.png")
```

---

## Implementation Plan

### Phase 1: Core Nodes (Week 1)

1. **Florence2BatchCaption node**
   - Batch processing of tiles
   - Caption generation with prepend/append
   - Output as both joined string and list

2. **PromptListToConditioning node**
   - Convert prompt list to CLIP embeddings
   - Handle batch encoding efficiently

### Phase 2: Integration (Week 2)

3. **Modify DynamicTileSplit**
   - Add tile position metadata to `tile_calc`
   - Pass through blend_mode selection

4. **TiledKSamplerWithPromptList node**
   - Per-tile conditioning support
   - Efficient batched processing

### Phase 3: Optimization (Week 3)

5. **Caching & Performance**
   - Cache Florence2 outputs for identical tiles
   - GPU memory optimization for large batches
   - Optional: parallel inference for Florence2

6. **Quality of Life**
   - Preview node showing tile + caption pairs
   - Caption editing node (manual override)
   - Batch caption export/import (JSON)

---

## File Structure

```
ComfyUI-Florence2-Tile-prompt/
├── __init__.py              # Node registration
├── nodes.py                 # Original Florence2 nodes (kijai)
├── tile_nodes.py            # NEW: Tile-specific nodes
│   ├── Florence2BatchCaption
│   ├── PromptListToConditioning
│   └── TiledKSamplerWithPromptList
├── blending/                # Import from SimpleTiles or include
│   ├── __init__.py
│   ├── linear.py
│   ├── noise.py
│   └── laplacian.py
├── utils.py                 # Shared utilities
│   ├── tensor_to_pil()
│   ├── pil_to_tensor()
│   └── batch_utils
├── configuration_florence2.py
├── modeling_florence2.py
├── requirements.txt
└── README.md
```

---

## Advanced Features (Future)

### Semantic Region Detection

Use Florence2's object detection to identify distinct regions before tiling:

```python
class Florence2SmartTileSplit:
    """
    Intelligent tile splitting based on semantic regions.
    Uses Florence2 detection to ensure tiles don't awkwardly 
    split important objects (faces, text, etc.)
    """
    # Uses <OD> or <DENSE_REGION_CAPTION> to find regions
    # Adjusts tile boundaries to respect object boundaries
```

### Cross-Tile Context

For better coherence, provide neighboring tile context:

```python
class Florence2BatchCaptionWithContext:
    """
    Caption each tile while providing neighboring tile context.
    Helps maintain consistency across tile boundaries.
    """
    def batch_caption_with_context(self, tiles, tile_calc, ...):
        # For each tile, include cropped neighbors in context
        # "This is the upper-left section. To the right: {neighbor_caption}"
```

### ControlNet Tile Integration

Combine with ControlNet Tile for even better results:

```python
# Use both Florence2 prompt AND ControlNet tile guidance
controlnet_apply = ControlNetApply(
    conditioning=per_tile_cond,
    control_net=tile_controlnet,
    image=tiles,
    strength=0.5
)
```

---

## Testing Checklist

- [ ] Florence2BatchCaption produces consistent captions
- [ ] Prompt list correctly maps to tile indices
- [ ] Per-tile conditioning affects output appropriately
- [ ] Blending modes (linear, noise, laplacian) work with processed tiles
- [ ] Memory usage acceptable for 16-tile batch
- [ ] Memory usage acceptable for 64-tile batch
- [ ] Edge cases: single tile, non-square tiles, extreme overlap
- [ ] Integration with existing SimpleTiles workflows
