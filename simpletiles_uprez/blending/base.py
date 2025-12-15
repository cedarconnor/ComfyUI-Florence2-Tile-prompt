"""
Abstract Base Class for Tile Blending Strategies.
"""

from abc import ABC, abstractmethod

import torch


class TileBlender(ABC):
    def __init__(self, blend_width: int):
        if blend_width < 0:
            raise ValueError(f"blend_width must be >= 0, got {blend_width}")

        self.blend_width = blend_width

    @abstractmethod
    def create_mask(self, height: int, width: int, direction: str, seed: int = 0) -> torch.Tensor:
        pass

    @abstractmethod
    def blend_tiles(self, canvas: torch.Tensor, tile: torch.Tensor, position: dict, tile_calc: dict) -> torch.Tensor:
        pass

    def blend_overlap_region(self, background: torch.Tensor, foreground: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
        return background * (1 - mask_expanded) + foreground * mask_expanded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(blend_width={self.blend_width})"

