"""
Vendored ComfyUI_SimpleTiles_Uprez

This package is vendored into ComfyUI-Florence2-Tile-prompt so the full
Florence2 per-tile prompting workflow can run without an external tiles
dependency.

Original project: `C:\\ComfyUI\\custom_nodes\\ComfyUI_SimpleTiles_Uprez`
"""

from .standard import TileSplit, TileMerge, TileCalc
from .dynamic import DynamicTileSplit, DynamicTileMerge

# Register nodes under unique names to avoid clashes with other tile packs.
NODE_CLASS_MAPPINGS = {
    "SimpleTilesUprezTileSplit": TileSplit,
    "SimpleTilesUprezTileMerge": TileMerge,
    "SimpleTilesUprezTileCalc": TileCalc,
    "SimpleTilesUprezDynamicTileSplit": DynamicTileSplit,
    "SimpleTilesUprezDynamicTileMerge": DynamicTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleTilesUprezTileSplit": "TileSplit (SimpleTiles Uprez Legacy)",
    "SimpleTilesUprezTileMerge": "TileMerge (SimpleTiles Uprez Legacy)",
    "SimpleTilesUprezTileCalc": "TileCalc (SimpleTiles Uprez Legacy)",
    "SimpleTilesUprezDynamicTileSplit": "TileSplit (SimpleTiles Uprez Dynamic)",
    "SimpleTilesUprezDynamicTileMerge": "TileMerge (SimpleTiles Uprez Dynamic)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TileSplit",
    "TileMerge",
    "TileCalc",
    "DynamicTileSplit",
    "DynamicTileMerge",
]

