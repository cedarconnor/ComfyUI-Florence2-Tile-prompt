"""
Vendored ComfyUI_SimpleTiles_Uprez

This package is vendored into ComfyUI-Florence2-Tile-prompt so the full
Florence2 per-tile prompting workflow can run without an external tiles
dependency.

Features:
- Multiple blending modes: linear, gaussian, cosine, noise, laplacian, accumulation
- Feather percentage control for intuitive blend width specification
- Edge-fallback strategy to prevent artifacts at image boundaries
- Accumulation buffer blending for order-independent results
- VRAM-conscious sequential processing mode
- Optimal tile size calculator with VRAM budget constraints
- Tile boundary preview for debugging

Original project: `C:\\ComfyUI\\custom_nodes\\ComfyUI_SimpleTiles_Uprez`
"""

from .standard import TileSplit, TileMerge, TileCalc
from .dynamic import (
    DynamicTileSplit,
    DynamicTileMerge,
    OptimalTileSizeCalculator,
    TileBoundaryPreview,
    UpscaleAwareTileMerge,
)

# Register nodes under unique names to avoid clashes with other tile packs.
NODE_CLASS_MAPPINGS = {
    # Legacy nodes
    "SimpleTilesUprezTileSplit": TileSplit,
    "SimpleTilesUprezTileMerge": TileMerge,
    "SimpleTilesUprezTileCalc": TileCalc,
    # Dynamic nodes
    "SimpleTilesUprezDynamicTileSplit": DynamicTileSplit,
    "SimpleTilesUprezDynamicTileMerge": DynamicTileMerge,
    # New utility nodes
    "SimpleTilesUprezOptimalTileSizeCalculator": OptimalTileSizeCalculator,
    "SimpleTilesUprezTileBoundaryPreview": TileBoundaryPreview,
    "SimpleTilesUprezUpscaleAwareTileMerge": UpscaleAwareTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Legacy nodes
    "SimpleTilesUprezTileSplit": "TileSplit (SimpleTiles Uprez Legacy)",
    "SimpleTilesUprezTileMerge": "TileMerge (SimpleTiles Uprez Legacy)",
    "SimpleTilesUprezTileCalc": "TileCalc (SimpleTiles Uprez Legacy)",
    # Dynamic nodes
    "SimpleTilesUprezDynamicTileSplit": "TileSplit (SimpleTiles Uprez Dynamic)",
    "SimpleTilesUprezDynamicTileMerge": "TileMerge (SimpleTiles Uprez Dynamic)",
    # New utility nodes
    "SimpleTilesUprezOptimalTileSizeCalculator": "Optimal Tile Size Calculator",
    "SimpleTilesUprezTileBoundaryPreview": "Tile Boundary Preview",
    "SimpleTilesUprezUpscaleAwareTileMerge": "Upscale-Aware Tile Merge",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    # Standard/Legacy
    "TileSplit",
    "TileMerge",
    "TileCalc",
    # Dynamic
    "DynamicTileSplit",
    "DynamicTileMerge",
    # New utilities
    "OptimalTileSizeCalculator",
    "TileBoundaryPreview",
    "UpscaleAwareTileMerge",
]

