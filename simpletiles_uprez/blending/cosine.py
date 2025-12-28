"""
Cosine Interpolation Blending.

Uses smooth S-curve cosine transitions for natural-looking seams.
The cosine curve provides smooth acceleration/deceleration at blend edges.
"""

import torch
import math

from .base import TileBlender


class CosineBlender(TileBlender):
    def __init__(self, blend_width: int):
        """
        Initialize Cosine blender.

        Args:
            blend_width: Width of the blend region in pixels
        """
        super().__init__(blend_width)

    def create_mask(self, height: int, width: int, direction: str, seed: int = 0) -> torch.Tensor:
        """
        Create cosine-interpolated blend mask.

        Uses the smoothstep formula based on cosine:
        f(t) = (1 - cos(t * pi)) / 2

        This creates an S-curve that starts slow, accelerates in the middle,
        and decelerates at the end - ideal for smooth blending.
        """
        if direction == "horizontal":
            t = torch.linspace(0, 1, width)
            # Cosine interpolation: smooth S-curve from 0 to 1
            mask = (1.0 - torch.cos(t * math.pi)) / 2.0
            mask = mask.unsqueeze(0).expand(height, -1)
        elif direction == "vertical":
            t = torch.linspace(0, 1, height)
            mask = (1.0 - torch.cos(t * math.pi)) / 2.0
            mask = mask.unsqueeze(1).expand(-1, width)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'")

        return mask

    def create_smooth_corner_mask(self, size: int) -> torch.Tensor:
        """
        Create a smooth corner blend mask using radial cosine interpolation.
        """
        y = torch.linspace(0, 1, size)
        x = torch.linspace(0, 1, size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Radial distance from corner (0,0)
        dist = torch.sqrt(xx ** 2 + yy ** 2) / math.sqrt(2)
        dist = torch.clamp(dist, 0, 1)

        # Apply cosine interpolation
        mask = (1.0 - torch.cos(dist * math.pi)) / 2.0

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

        # Blend left overlap with cosine curve
        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal")
            bg = canvas[:, py : py + th, px : px + blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py : py + th, px : px + blend_w, :] = self.blend_overlap_region(bg, fg, mask)

        # Blend top overlap with cosine curve
        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical")
            bg = canvas[:, py : py + blend_w, px : px + tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py : py + blend_w, px : px + tw, :] = self.blend_overlap_region(bg, fg, mask)

        # Blend corner with smooth radial cosine
        if has_left and has_top and blend_w > 0:
            # Use combination of horizontal and vertical for consistent behavior
            mask_h = self.create_mask(blend_w, blend_w, "horizontal")
            mask_v = self.create_mask(blend_w, blend_w, "vertical")
            # Average for smooth corner (cosine curves combine nicely)
            mask_corner = (mask_h + mask_v) / 2.0

            bg = canvas[:, py : py + blend_w, px : px + blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py : py + blend_w, px : px + blend_w, :] = self.blend_overlap_region(bg, fg, mask_corner)

        return result

    def __repr__(self) -> str:
        return f"CosineBlender(blend_width={self.blend_width})"
