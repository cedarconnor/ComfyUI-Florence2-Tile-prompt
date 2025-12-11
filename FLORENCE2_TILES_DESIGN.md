# Florence2 Tile Prompt + SimpleTiles Integration

## Overview

This design integrates **per-tile VLM captioning** using Florence2 with the **advanced blending** capabilities of SimpleTiles_Uprez. The goal is to enable intelligent upscaling where each tile receives its own context-aware prompt, then gets blended back seamlessly.

**Status**: Core functionality implemented. Advanced features listed below are planned for future releases.

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

## Implementation Status

### ✅ Implemented (v1.0.0)

The following nodes are fully implemented and ready to use:

1. **Florence2BatchCaption** - Processes batched tiles through Florence2 for per-tile captioning
2. **PromptListToConditioning** - Converts prompt lists to CLIP conditioning for per-tile use
3. **TiledSamplerWithPromptList** - Samples each tile with individual conditioning
4. **TilePromptPreview** - Displays tiles alongside their generated prompts
5. **PromptListEditor** - Allows manual override of individual tile prompts
6. **TileCalcAddPositions** - Augments tile_calc with position metadata

### 🔄 Planned Features (Future)

- **Semantic Region Detection** - Smart tile splitting based on object detection
- **Cross-Tile Context** - Provide neighboring tile context for better coherence
- **ControlNet Tile Integration** - Combine Florence2 prompts with ControlNet guidance
- **Caption Caching** - Cache Florence2 outputs for identical tiles
- **Parallel Inference** - Process multiple tiles simultaneously
- **Batch Caption Export/Import** - Save/load captions as JSON

---

## Implemented Nodes

### 1. Florence2BatchCaption

**Status**: ✅ Implemented

Processes a batch of tile images through Florence2 and returns an array of prompts.

**Key Features**:
- Supports multiple caption modes (caption, detailed_caption, more_detailed_caption)
- Customizable prepend/append text for prompt formatting
- Optional global context for better captions
- Position-aware captioning when tile_calc metadata is available
- Progress bar for long operations
- Robust error handling and input validation

**Returns**:
- `prompts_preview`: Joined string preview of all prompts
- `prompts_list`: List of individual prompts (one per tile)

**Example Usage**:
See "Complete Workflow Example" section below for integration with other nodes.

### 2. PromptListToConditioning

**Status**: ✅ Implemented

Converts a list of prompts into per-tile CONDITIONING that can be used with batch processing.

**Key Features**:
- Accepts prompt lists from Florence2BatchCaption
- Encodes each prompt with CLIP model
- Returns list of conditioning (one per tile)
- Handles nested list inputs automatically

### 3. TiledSamplerWithPromptList

**Status**: ✅ Implemented

A modified KSampler that accepts per-tile conditioning for context-aware tile processing.

**Key Features**:
- Processes each tile with individual positive conditioning
- Falls back to global conditioning when per-tile not available
- Per-tile seed variation for natural results
- Progress bar for batch operations
- Comprehensive input validation
- Handles RGB channel extraction automatically
- Robust VAE encoding/decoding with error checking

---

## Integration with Existing SimpleTiles

This implementation works with the existing SimpleTiles_Uprez nodes without requiring modifications:

### DynamicTileSplit (SimpleTiles)

**No modifications required**. The existing node provides:
- Batched tile output (IMAGE)
- tile_calc metadata with overlap, grid_size, original_size

### TileCalcAddPositions (This Package)

Use this helper node if your SimpleTiles version doesn't include tile positions:
- Infers tile positions from grid layout
- Adds `tile_positions` list to tile_calc
- Required for position-aware Florence2 captioning

### DynamicTileMerge (SimpleTiles)

**No modifications required**. The existing node:
- Accepts batched processed tiles
- Uses tile_calc for seamless blending
- Supports multiple blend modes (linear, noise, laplacian)

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

## Implementation Status Summary

### ✅ Completed (v1.0.0)

**Core Functionality**:
- ✅ Florence2BatchCaption - Batch processing with progress bars
- ✅ PromptListToConditioning - CLIP encoding for per-tile conditioning
- ✅ TiledSamplerWithPromptList - Per-tile conditional sampling
- ✅ TilePromptPreview - Preview tiles with captions
- ✅ PromptListEditor - Manual prompt override
- ✅ TileCalcAddPositions - Position metadata injection

**Quality & Robustness**:
- ✅ Comprehensive input validation
- ✅ Progress bars for long operations
- ✅ Detailed error messages with context
- ✅ Proper dtype handling for mixed precision models
- ✅ RGB channel extraction for VAE encoding

### 🔄 Future Enhancements

**Performance Optimizations**:
- Caption caching for identical tiles
- Parallel Florence2 inference
- GPU memory optimization for large batches

**Advanced Features**:
- Semantic region detection for smart tiling
- Cross-tile context awareness
- ControlNet integration
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
