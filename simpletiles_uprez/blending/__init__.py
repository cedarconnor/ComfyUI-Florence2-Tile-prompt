"""
Advanced Tile Blending Module for SimpleTiles Uprez.
"""

from .base import TileBlender
from .linear import LinearBlender
from .noise import NoiseBlender
from .laplacian import LaplacianBlender

BLENDERS = {
    "linear": LinearBlender,
    "noise": NoiseBlender,
    "laplacian": LaplacianBlender,
}


def get_blender(mode: str, blend_width: int, **kwargs):
    if mode not in BLENDERS:
        available = list(BLENDERS.keys())
        raise ValueError(f"Unknown blend mode: '{mode}'. Available modes: {available}")

    return BLENDERS[mode](blend_width, **kwargs)


__all__ = [
    "TileBlender",
    "get_blender",
    "LinearBlender",
    "NoiseBlender",
    "LaplacianBlender",
]

