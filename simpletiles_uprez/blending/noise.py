"""
Noise-Based Blending.
"""

import torch

from .base import TileBlender


class NoiseBlender(TileBlender):
    def __init__(self, blend_width: int, frequency: float = 0.05, octaves: int = 3):
        super().__init__(blend_width)

        if frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {frequency}")
        if octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {octaves}")

        self.frequency = frequency
        self.octaves = octaves

    def _smoothstep(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * (3 - 2 * t)

    def _perlin_1d(self, x: torch.Tensor, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed)

        x0 = x.floor().long()
        x1 = x0 + 1

        t = x - x0.float()
        t = self._smoothstep(t)

        g0 = torch.rand_like(x0.float()) * 2 - 1
        g1 = torch.rand_like(x1.float()) * 2 - 1

        d0 = x - x0.float()
        d1 = x - x1.float()

        v0 = g0 * d0
        v1 = g1 * d1

        return v0 * (1 - t) + v1 * t

    def _fbm_1d(self, x: torch.Tensor, seed: int = 0) -> torch.Tensor:
        result = torch.zeros_like(x)

        amplitude = 1.0
        frequency = self.frequency

        for octave in range(self.octaves):
            result += amplitude * self._perlin_1d(x * frequency, seed=seed + octave * 1000)
            amplitude *= 0.5
            frequency *= 2.0

        return result

    def create_mask(self, height: int, width: int, direction: str, seed: int = 0) -> torch.Tensor:
        blend_w = width if direction == "horizontal" else height
        blend_w = max(1, blend_w)

        noise_scale = blend_w * 0.3
        softness = max(1.0, blend_w * 0.15)

        if direction == "horizontal":
            x = torch.linspace(0, height - 1, height)
            noise = self._fbm_1d(x, seed=seed)
            noise = noise / (noise.abs().max() + 1e-8)

            boundary = blend_w * 0.5 + noise * noise_scale
            boundary = boundary.clamp(0, blend_w)

            y = torch.linspace(0, width - 1, width)
            y = y.unsqueeze(0).expand(height, -1)
            boundary = boundary.unsqueeze(1).expand(-1, width)

            dist = (y - boundary) / softness
            mask = torch.sigmoid(dist)

        elif direction == "vertical":
            x = torch.linspace(0, width - 1, width)
            noise = self._fbm_1d(x, seed=seed)
            noise = noise / (noise.abs().max() + 1e-8)

            boundary = blend_w * 0.5 + noise * noise_scale
            boundary = boundary.clamp(0, blend_w)

            y = torch.linspace(0, height - 1, height)
            y = y.unsqueeze(1).expand(-1, width)
            boundary = boundary.unsqueeze(0).expand(height, -1)

            dist = (y - boundary) / softness
            mask = torch.sigmoid(dist)

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

        idx = position.get("index", 0)
        base_seed = idx * 12345

        result = canvas.clone()

        cx1 = blend_w if has_left else 0
        cy1 = blend_w if has_top else 0

        if cy1 < th and cx1 < tw:
            result[:, py + cy1 : py + th, px + cx1 : px + tw, :] = tile[:, cy1:, cx1:, :]

        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal", seed=base_seed)
            bg = canvas[:, py : py + th, px : px + blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py : py + th, px : px + blend_w, :] = self.blend_overlap_region(bg, fg, mask)

        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical", seed=base_seed + 1)
            bg = canvas[:, py : py + blend_w, px : px + tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py : py + blend_w, px : px + tw, :] = self.blend_overlap_region(bg, fg, mask)

        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal", seed=base_seed)
            mask_v = self.create_mask(blend_w, blend_w, "vertical", seed=base_seed + 1)
            mask_corner = mask_h * mask_v

            bg = canvas[:, py : py + blend_w, px : px + blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py : py + blend_w, px : px + blend_w, :] = self.blend_overlap_region(bg, fg, mask_corner)

        return result

    def __repr__(self) -> str:
        return (
            f"NoiseBlender(blend_width={self.blend_width}, "
            f"frequency={self.frequency}, octaves={self.octaves})"
        )

