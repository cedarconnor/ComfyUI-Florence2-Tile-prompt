"""
Laplacian Pyramid Blending.
"""

import torch
import torch.nn.functional as F

from .base import TileBlender


class LaplacianBlender(TileBlender):
    def __init__(self, blend_width: int, levels: int = 4):
        super().__init__(blend_width)

        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")

        self.levels = levels
        self._gaussian_kernel_cache = {}

    def _gaussian_kernel(self, size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        cache_key = (size, sigma)
        if cache_key in self._gaussian_kernel_cache:
            return self._gaussian_kernel_cache[cache_key]

        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = torch.outer(g, g)
        self._gaussian_kernel_cache[cache_key] = kernel
        return kernel

    def _downsample(self, img: torch.Tensor) -> torch.Tensor:
        x = img.permute(0, 3, 1, 2)

        kernel = self._gaussian_kernel(5, 1.0).to(x.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)

        pad = kernel.shape[-1] // 2
        x = F.conv2d(x, kernel, padding=pad, groups=x.shape[1])

        x = x[:, :, ::2, ::2]
        return x.permute(0, 2, 3, 1)

    def _upsample(self, img: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = img.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1)

    def _build_gaussian_pyramid(self, img: torch.Tensor) -> list[torch.Tensor]:
        pyr = [img]
        current = img

        for _ in range(self.levels - 1):
            current = self._downsample(current)
            pyr.append(current)

        return pyr

    def _build_laplacian_pyramid(self, img: torch.Tensor) -> list[torch.Tensor]:
        gaussian = self._build_gaussian_pyramid(img)
        laplacian = []

        for i in range(len(gaussian) - 1):
            current = gaussian[i]
            next_level = gaussian[i + 1]
            up = self._upsample(next_level, (current.shape[1], current.shape[2]))
            laplacian.append(current - up)

        laplacian.append(gaussian[-1])
        return laplacian

    def _collapse_laplacian_pyramid(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        current = pyramid[-1]

        for level in reversed(pyramid[:-1]):
            current = self._upsample(current, (level.shape[1], level.shape[2]))
            current = current + level

        return current

    def _build_mask_pyramid(self, mask: torch.Tensor) -> list[torch.Tensor]:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)

        mask = mask.float()
        pyr = [mask]
        current = mask

        for _ in range(self.levels - 1):
            current = self._downsample(current)
            pyr.append(current)

        return pyr

    def create_mask(self, height: int, width: int, direction: str, seed: int = 0) -> torch.Tensor:
        if direction == "horizontal":
            mask = torch.linspace(0, 1, width).unsqueeze(0).expand(height, -1)
        elif direction == "vertical":
            mask = torch.linspace(0, 1, height).unsqueeze(1).expand(-1, width)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'")

        return mask

    def blend_region_laplacian(self, background: torch.Tensor, foreground: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lap_bg = self._build_laplacian_pyramid(background)
        lap_fg = self._build_laplacian_pyramid(foreground)
        mask_pyr = self._build_mask_pyramid(mask.to(background.device))

        blended_pyramid = []
        for l_bg, l_fg, m in zip(lap_bg, lap_fg, mask_pyr):
            if m.shape[1:3] != l_bg.shape[1:3]:
                m_temp = m.permute(0, 3, 1, 2)
                m_temp = F.interpolate(
                    m_temp,
                    size=(l_bg.shape[1], l_bg.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                )
                m = m_temp.permute(0, 2, 3, 1)

            blended = l_bg * (1 - m) + l_fg * m
            blended_pyramid.append(blended)

        result = self._collapse_laplacian_pyramid(blended_pyramid)
        return result.clamp(0.0, 1.0)

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
            result[:, py : py + th, px : px + blend_w, :] = self.blend_region_laplacian(bg, fg, mask)

        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical")
            bg = canvas[:, py : py + blend_w, px : px + tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py : py + blend_w, px : px + tw, :] = self.blend_region_laplacian(bg, fg, mask)

        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal")
            mask_v = self.create_mask(blend_w, blend_w, "vertical")
            mask_corner = (mask_h + mask_v) / 2

            bg = canvas[:, py : py + blend_w, px : px + blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py : py + blend_w, px : px + blend_w, :] = self.blend_region_laplacian(bg, fg, mask_corner)

        return result

