"""
Advanced Tile Blending Module for SimpleTiles Uprez.

Available blending modes:
- linear: Simple linear gradient blending (fastest)
- gaussian: Center-weighted Gaussian masks with smooth falloff
- cosine: S-curve cosine interpolation for natural transitions
- noise: Perlin noise-based organic blending
- laplacian: Multi-scale Laplacian pyramid blending (highest quality)
- accumulation: Accumulation buffer weighted averaging (order-independent)
"""

from .base import TileBlender
from .linear import LinearBlender
from .gaussian import GaussianBlender
from .cosine import CosineBlender
from .noise import NoiseBlender
from .laplacian import LaplacianBlender
from .accumulation import AccumulationBlender, AccumulationTileMerger

BLENDERS = {
    "linear": LinearBlender,
    "gaussian": GaussianBlender,
    "cosine": CosineBlender,
    "noise": NoiseBlender,
    "laplacian": LaplacianBlender,
    "accumulation": AccumulationBlender,
}


def get_blender(mode: str, blend_width: int, **kwargs):
    """
    Factory function to create a blender instance.

    Args:
        mode: Blending algorithm name (linear, gaussian, cosine, noise, laplacian)
        blend_width: Width of blend region in pixels
        **kwargs: Additional arguments passed to specific blenders
            - gaussian: sigma_factor (0.1-0.5, controls spread)
            - noise: frequency, octaves
            - laplacian: levels

    Returns:
        TileBlender instance
    """
    if mode not in BLENDERS:
        available = list(BLENDERS.keys())
        raise ValueError(f"Unknown blend mode: '{mode}'. Available modes: {available}")

    return BLENDERS[mode](blend_width, **kwargs)


def get_available_blend_modes() -> list:
    """Return list of available blend mode names."""
    return list(BLENDERS.keys())


__all__ = [
    "TileBlender",
    "get_blender",
    "get_available_blend_modes",
    "LinearBlender",
    "GaussianBlender",
    "CosineBlender",
    "NoiseBlender",
    "LaplacianBlender",
    "AccumulationBlender",
    "AccumulationTileMerger",
]

