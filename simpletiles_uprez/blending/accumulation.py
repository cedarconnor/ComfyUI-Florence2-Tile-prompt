"""
Accumulation Buffer Weighted Averaging Blender.

This blender uses a separate weight accumulation buffer to avoid
order-dependent blending artifacts. All tiles contribute to a weighted
sum, and the final result is normalized by the accumulated weights.

Formula: Final_Pixel = Sum(Tile × Weight) / (Sum(Weight) + ε)

This approach is more sophisticated than sequential blending as it:
1. Eliminates order-dependent artifacts
2. Handles overlapping regions more accurately
3. Provides consistent results regardless of tile processing order
"""

import torch
import math

from .base import TileBlender


class AccumulationBlender(TileBlender):
    """
    Accumulation buffer blender that collects weighted contributions
    from all tiles before normalizing to produce the final image.
    """

    def __init__(self, blend_width: int, weight_mode: str = "gaussian", edge_fallback: bool = True):
        """
        Initialize accumulation blender.

        Args:
            blend_width: Width of the blend/feather region in pixels
            weight_mode: Weight distribution mode ('gaussian', 'cosine', 'linear')
            edge_fallback: Apply edge-fallback strategy to prevent artifacts at boundaries
        """
        super().__init__(blend_width)
        self.weight_mode = weight_mode
        self.edge_fallback = edge_fallback
        self._epsilon = 1e-8

    def create_tile_weight_mask(
        self,
        height: int,
        width: int,
        row: int,
        col: int,
        total_rows: int,
        total_cols: int,
        feather: int
    ) -> torch.Tensor:
        """
        Create a weight mask for a tile based on its position in the grid.

        Tiles at edges have asymmetric weights (no falloff at image boundaries).
        Interior tiles have symmetric falloff on all overlapping edges.

        Args:
            height: Tile height
            width: Tile width
            row: Current tile row index
            col: Current tile column index
            total_rows: Total rows in grid
            total_cols: Total columns in grid
            feather: Feather/blend width in pixels

        Returns:
            2D weight mask tensor
        """
        mask = torch.ones(height, width)

        feather = min(feather, height // 2, width // 2)
        if feather <= 0:
            return mask

        # Determine which edges need falloff (not at image boundaries if edge_fallback)
        has_left = col > 0
        has_right = col < total_cols - 1
        has_top = row > 0
        has_bottom = row < total_rows - 1

        # Create weight gradients based on mode
        if self.weight_mode == "gaussian":
            t = torch.linspace(0, 1, feather)
            # Gaussian CDF-like curve
            weights = 1.0 - torch.exp(-3 * t ** 2)
        elif self.weight_mode == "cosine":
            t = torch.linspace(0, 1, feather)
            weights = (1.0 - torch.cos(t * math.pi)) / 2.0
        else:  # linear
            weights = torch.linspace(0, 1, feather)

        # Apply weights to edges that need falloff
        if has_left:
            for i in range(feather):
                mask[:, i] *= weights[i]

        if has_right:
            for i in range(feather):
                mask[:, -(i + 1)] *= weights[i]

        if has_top:
            for i in range(feather):
                mask[i, :] *= weights[i]

        if has_bottom:
            for i in range(feather):
                mask[-(i + 1), :] *= weights[i]

        return mask

    def create_mask(self, height: int, width: int, direction: str, seed: int = 0) -> torch.Tensor:
        """Create a simple directional mask for compatibility."""
        if direction == "horizontal":
            if self.weight_mode == "gaussian":
                t = torch.linspace(0, 1, width)
                mask = 1.0 - torch.exp(-3 * t ** 2)
            elif self.weight_mode == "cosine":
                t = torch.linspace(0, 1, width)
                mask = (1.0 - torch.cos(t * math.pi)) / 2.0
            else:
                mask = torch.linspace(0, 1, width)
            mask = mask.unsqueeze(0).expand(height, -1)
        elif direction == "vertical":
            if self.weight_mode == "gaussian":
                t = torch.linspace(0, 1, height)
                mask = 1.0 - torch.exp(-3 * t ** 2)
            elif self.weight_mode == "cosine":
                t = torch.linspace(0, 1, height)
                mask = (1.0 - torch.cos(t * math.pi)) / 2.0
            else:
                mask = torch.linspace(0, 1, height)
            mask = mask.unsqueeze(1).expand(-1, width)
        else:
            raise ValueError(f"Invalid direction: {direction}")

        return mask

    def blend_tiles(self, canvas: torch.Tensor, tile: torch.Tensor, position: dict, tile_calc: dict) -> torch.Tensor:
        """
        Standard sequential blend for compatibility.
        For true accumulation blending, use AccumulationTileMerger class.
        """
        # Fall back to weighted blending similar to other blenders
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
        rows = tile_calc.get("rows", 1)
        cols = tile_calc.get("cols", 1)

        # Create weight mask for this tile
        weight_mask = self.create_tile_weight_mask(
            th, tw, row, col, rows, cols, blend_w
        )
        weight_mask = weight_mask.unsqueeze(0).unsqueeze(-1).to(tile.device)

        result = canvas.clone()

        # Get existing canvas region
        canvas_region = canvas[:, py:py + th, px:px + tw, :]
        existing_weight = (canvas_region.abs().sum(dim=-1, keepdim=True) > 0).float()

        # Weighted blend
        combined_weight = existing_weight + weight_mask
        combined_weight = torch.clamp(combined_weight, min=self._epsilon)

        blended = (canvas_region * existing_weight + tile * weight_mask) / combined_weight
        result[:, py:py + th, px:px + tw, :] = blended

        return result

    def __repr__(self) -> str:
        return f"AccumulationBlender(blend_width={self.blend_width}, weight_mode='{self.weight_mode}', edge_fallback={self.edge_fallback})"


class AccumulationTileMerger:
    """
    Standalone merger that uses accumulation buffer approach.

    Usage:
        merger = AccumulationTileMerger(canvas_height, canvas_width, channels, blend_width)
        for tile, position in tiles_and_positions:
            merger.add_tile(tile, position, tile_calc)
        result = merger.finalize()
    """

    def __init__(
        self,
        height: int,
        width: int,
        channels: int = 3,
        blend_width: int = 64,
        weight_mode: str = "gaussian",
        edge_fallback: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the accumulation merger.

        Args:
            height: Canvas height
            width: Canvas width
            channels: Number of color channels
            blend_width: Feather width in pixels
            weight_mode: Weight distribution ('gaussian', 'cosine', 'linear')
            edge_fallback: Adjust weights at image boundaries
            device: Torch device
            dtype: Tensor dtype
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.blend_width = blend_width
        self.weight_mode = weight_mode
        self.edge_fallback = edge_fallback
        self.device = device
        self.dtype = dtype

        # Accumulation buffers
        self.color_buffer = torch.zeros(
            (1, height, width, channels), dtype=dtype, device=device
        )
        self.weight_buffer = torch.zeros(
            (1, height, width, 1), dtype=dtype, device=device
        )

        self._epsilon = 1e-8
        self._blender = AccumulationBlender(blend_width, weight_mode, edge_fallback)

    def add_tile(self, tile: torch.Tensor, position: dict, tile_calc: dict) -> None:
        """
        Add a tile's contribution to the accumulation buffers.

        Args:
            tile: Tile tensor (1, H, W, C)
            position: Position dict with row, col, place_x, place_y
            tile_calc: Tile calculation metadata with rows, cols
        """
        _, th, tw, _ = tile.shape

        if "place_x" in position and "place_y" in position:
            px = position["place_x"]
            py = position["place_y"]
        else:
            px = position.get("x1", 0)
            py = position.get("y1", 0)

        row = position.get("row", 0)
        col = position.get("col", 0)
        rows = tile_calc.get("rows", 1)
        cols = tile_calc.get("cols", 1)

        # Create weight mask
        weight_mask = self._blender.create_tile_weight_mask(
            th, tw, row, col, rows, cols, self.blend_width
        )
        weight_mask = weight_mask.unsqueeze(0).unsqueeze(-1).to(self.device, self.dtype)

        # Clamp placement to canvas bounds (edge fallback)
        py_end = min(py + th, self.height)
        px_end = min(px + tw, self.width)
        py = max(0, py)
        px = max(0, px)

        # Adjust tile slice if placement was clamped
        tile_y_start = 0 if py >= 0 else -py
        tile_x_start = 0 if px >= 0 else -px
        tile_y_end = th - max(0, (py + th) - self.height)
        tile_x_end = tw - max(0, (px + tw) - self.width)

        tile_slice = tile[:, tile_y_start:tile_y_end, tile_x_start:tile_x_end, :]
        weight_slice = weight_mask[:, tile_y_start:tile_y_end, tile_x_start:tile_x_end, :]

        # Accumulate
        self.color_buffer[:, py:py_end, px:px_end, :] += tile_slice * weight_slice
        self.weight_buffer[:, py:py_end, px:px_end, :] += weight_slice

    def finalize(self) -> torch.Tensor:
        """
        Normalize the accumulated buffer to produce the final image.

        Returns:
            Final merged image tensor (1, H, W, C)
        """
        result = self.color_buffer / (self.weight_buffer + self._epsilon)
        return result.clamp(0.0, 1.0)

    def reset(self) -> None:
        """Clear the accumulation buffers."""
        self.color_buffer.zero_()
        self.weight_buffer.zero_()
