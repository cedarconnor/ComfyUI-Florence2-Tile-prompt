"""
ComfyUI-Florence2-Tile-prompt

Combines Florence2 VLM captioning with SimpleTiles for intelligent
per-tile prompting during upscaling workflows.

Integration Architecture:
1. Split image into tiles (DynamicTileSplit from SimpleTiles)
2. Caption each tile with Florence2 (Florence2BatchCaption)
3. Process tiles with per-tile prompts (TiledSamplerWithPromptList)
4. Merge with advanced blending (DynamicTileMerge from SimpleTiles)
"""

# Import original Florence2 nodes
from .nodes import (
    NODE_CLASS_MAPPINGS as FLORENCE2_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FLORENCE2_NODE_DISPLAY_NAME_MAPPINGS,
)

# Import tile-specific nodes
from .tile_nodes import (
    NODE_CLASS_MAPPINGS as TILE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as TILE_NODE_DISPLAY_NAME_MAPPINGS,
)

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **FLORENCE2_NODE_CLASS_MAPPINGS,
    **TILE_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FLORENCE2_NODE_DISPLAY_NAME_MAPPINGS,
    **TILE_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.0.0"
__author__ = "Cedar"
