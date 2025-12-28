"""
Gaussian Weighted Blending.

Uses center-weighted Gaussian masks for smooth seam hiding.
Weights decay exponentially from tile center toward edges.
"""

import torch
import math

from .base import TileBlender


class GaussianBlender(TileBlender):
    def __init__(self, blend_width: int, sigma_factor: float = 0.3):
        """
        Initialize Gaussian blender.

        Args:
            blend_width: Width of the blend region in pixels
            sigma_factor: Controls Gaussian spread (0.1-0.5, lower = sharper falloff)
        """
        super().__init__(blend_width)
        self.sigma_factor = max(0.1, min(0.5, sigma_factor))

    def _gaussian_1d(self, size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel centered at the middle."""
        x = torch.arange(size).float()
        center = (size - 1) / 2.0
        return torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    def create_mask(self, height: int, width: int, direction: str, seed: int = 0) -> torch.Tensor:
        """
        Create Gaussian-weighted blend mask.

        For horizontal blending, creates a gradient from 0 to 1 across width
        with Gaussian smoothing. For vertical, across height.
        """
        if direction == "horizontal":
            # Gaussian curve from 0 to 1 across width
            sigma = width * self.sigma_factor
            x = torch.linspace(0, 1, width)
            # Apply sigmoid-like Gaussian curve for smooth transition
            center = 0.5
            mask = 1.0 / (1.0 + torch.exp(-((x - center) / (self.sigma_factor / 2))))
            mask = mask.unsqueeze(0).expand(height, -1)
        elif direction == "vertical":
            sigma = height * self.sigma_factor
            y = torch.linspace(0, 1, height)
            center = 0.5
            mask = 1.0 / (1.0 + torch.exp(-((y - center) / (self.sigma_factor / 2))))
            mask = mask.unsqueeze(1).expand(-1, width)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'")

        return mask

    def create_center_weight_mask(self, height: int, width: int) -> torch.Tensor:
        """
        Create a 2D Gaussian weight mask centered in the tile.
        Maximum weight at center, decaying toward edges.
        """
        sigma_y = height * self.sigma_factor
        sigma_x = width * self.sigma_factor

        y = torch.arange(height).float()
        x = torch.arange(width).float()

        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0

        gauss_y = torch.exp(-((y - center_y) ** 2) / (2 * sigma_y ** 2))
        gauss_x = torch.exp(-((x - center_x) ** 2) / (2 * sigma_x ** 2))

        # Outer product for 2D Gaussian
        mask = torch.outer(gauss_y, gauss_x)

        return mask

    def blend_tiles(self, canvas: torch.Tensor, tile: torch.Tensor, position: dict, tile_calc: dict) -> torch.Tensor:
        blend_w = self.blend_width

        _, th, tw, _ = tile.shape

        if "place_x" in position and "place_y" in position:
            px = position["place_x"]
            py = position["place_y"]
        else:
            px = position.get("x1", 0)
            py = position.get("y1", 0)

        row = position.get("row", 0)
        col = position.get("col", 0)
        has_left = col > 0
        has_top = row > 0

        blend_w = min(blend_w, th, tw)

        result = canvas.clone()

        cx1 = blend_w if has_left else 0
        cy1 = blend_w if has_top else 0

        # Place non-overlapping region directly
        if cy1 < th and cx1 < tw:
            result[:, py + cy1 : py + th, px + cx1 : px + tw, :] = tile[:, cy1:, cx1:, :]

        # Blend left overlap with Gaussian weights
        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal")
            bg = canvas[:, py : py + th, px : px + blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py : py + th, px : px + blend_w, :] = self.blend_overlap_region(bg, fg, mask)

        # Blend top overlap with Gaussian weights
        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical")
            bg = canvas[:, py : py + blend_w, px : px + tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py : py + blend_w, px : px + tw, :] = self.blend_overlap_region(bg, fg, mask)

        # Blend corner with combined Gaussian weights
        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal")
            mask_v = self.create_mask(blend_w, blend_w, "vertical")
            # Combine using multiplication for smooth corner transition
            mask_corner = mask_h * mask_v

            bg = canvas[:, py : py + blend_w, px : px + blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py : py + blend_w, px : px + blend_w, :] = self.blend_overlap_region(bg, fg, mask_corner)

        return result

    def __repr__(self) -> str:
        return f"GaussianBlender(blend_width={self.blend_width}, sigma_factor={self.sigma_factor})"
