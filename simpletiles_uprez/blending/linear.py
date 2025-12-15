"""
Linear Gradient Blending.
"""

import torch

from .base import TileBlender


class LinearBlender(TileBlender):
    def create_mask(self, height: int, width: int, direction: str, seed: int = 0) -> torch.Tensor:
        if direction == "horizontal":
            mask = torch.linspace(0, 1, width)
            mask = mask.unsqueeze(0).expand(height, -1)
        elif direction == "vertical":
            mask = torch.linspace(0, 1, height)
            mask = mask.unsqueeze(1).expand(-1, width)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'")

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

        if cy1 < th and cx1 < tw:
            result[:, py + cy1 : py + th, px + cx1 : px + tw, :] = tile[:, cy1:, cx1:, :]

        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal")
            bg = canvas[:, py : py + th, px : px + blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py : py + th, px : px + blend_w, :] = self.blend_overlap_region(bg, fg, mask)

        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical")
            bg = canvas[:, py : py + blend_w, px : px + tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py : py + blend_w, px : px + tw, :] = self.blend_overlap_region(bg, fg, mask)

        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal")
            mask_v = self.create_mask(blend_w, blend_w, "vertical")
            mask_corner = (mask_h + mask_v) / 2

            bg = canvas[:, py : py + blend_w, px : px + blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py : py + blend_w, px : px + blend_w, :] = self.blend_overlap_region(bg, fg, mask_corner)

        return result

