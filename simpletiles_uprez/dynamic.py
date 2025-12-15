import os
import re
from datetime import datetime

import torch
from PIL import Image

import folder_paths

from .blending import get_blender


def order_by_center_last(tiles, image_width, image_height, tile_width, tile_height):
    # Order the tiles so the center is merged last. This ensures the center area
    # is the most refined in workflows that refine tiles iteratively.
    center_x = image_width // 2
    center_y = image_height // 2

    tiles = sorted(
        tiles,
        key=lambda tile: abs(tile[0] + tile_width // 2 - center_x)
        + abs(tile[1] + tile_height // 2 - center_y),
    )

    return tiles[::-1]


def generate_tiles(image_width, image_height, tile_width, tile_height, overlap, offset=0):
    tiles = []

    y = 0
    while y < image_height:
        if y == 0:
            next_y = y + tile_height - overlap + offset
        else:
            next_y = y + tile_height - overlap

        if y + tile_height >= image_height:
            y = max(image_height - tile_height, 0)
            next_y = image_height

        x = 0
        while x < image_width:
            if x == 0:
                next_x = x + tile_width - overlap + offset
            else:
                next_x = x + tile_width - overlap
            if x + tile_width >= image_width:
                x = max(image_width - tile_width, 0)
                next_x = image_width

            tiles.append((x, y))

            if next_x > image_width:
                break
            x = next_x

        if next_y > image_height:
            break
        y = next_y

    return order_by_center_last(tiles, image_width, image_height, tile_width, tile_height)


class DynamicTileSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image tensor to split into tiles."}),
                "tile_width": ("INT", {"default": 512, "min": 1, "max": 10000, "tooltip": "Tile width in pixels before upscaling."}),
                "tile_height": ("INT", {"default": 512, "min": 1, "max": 10000, "tooltip": "Tile height in pixels before upscaling."}),
                "overlap": ("INT", {"default": 128, "min": 1, "max": 10000, "tooltip": "Overlap between adjacent tiles in pixels."}),
                "blend_mode": (["linear", "noise", "laplacian"], {"default": "linear", "tooltip": "Blending algorithm: linear (fast), noise (balanced), laplacian (best quality)."}),
                "offset": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Initial offset applied to the first tile row and column."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_CALC")
    FUNCTION = "process"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def process(self, image, tile_width, tile_height, overlap, blend_mode, offset):
        image_height = image.shape[1]
        image_width = image.shape[2]

        tile_coordinates = generate_tiles(
            image_width, image_height, tile_width, tile_height, overlap, offset
        )

        image_tiles = []
        tile_positions = []

        # Calculate grid dimensions
        rows = 0
        cols = 0
        for coord in tile_coordinates:
            x, y = coord
            row = y // (tile_height - overlap) if overlap < tile_height else 0
            col = x // (tile_width - overlap) if overlap < tile_width else 0
            rows = max(rows, row + 1)
            cols = max(cols, col + 1)

        for idx, tile_coordinate in enumerate(tile_coordinates):
            x_start = max(0, tile_coordinate[0])
            y_start = max(0, tile_coordinate[1])
            x_end = min(image_width, x_start + tile_width)
            y_end = min(image_height, y_start + tile_height)

            image_tile = image[:, y_start:y_end, x_start:x_end, :]
            image_tiles.append(image_tile)

            row = y_start // (tile_height - overlap) if overlap < tile_height else 0
            col = x_start // (tile_width - overlap) if overlap < tile_width else 0

            tile_positions.append(
                {
                    "index": idx,
                    "row": row,
                    "col": col,
                    "x1": x_start,
                    "y1": y_start,
                    "x2": x_end,
                    "y2": y_end,
                    "place_x": x_start,
                    "place_y": y_start,
                }
            )

        tiles_tensor = torch.stack(image_tiles).squeeze(1)

        tile_calc = {
            "overlap": overlap,
            "overlap_x": overlap,
            "overlap_y": overlap,
            "image_height": image_height,
            "image_width": image_width,
            "offset": offset,
            "tile_height": tile_height,
            "tile_width": tile_width,
            "rows": rows,
            "cols": cols,
            "grid_size": (rows, cols),
            "blend_mode": blend_mode,
            "tile_positions": tile_positions,
        }

        return (tiles_tensor, tile_calc)


class DynamicTileMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of tiles to merge back into the full image."}),
                "blend": ("INT", {"default": 64, "min": 0, "max": 4096, "tooltip": "Number of pixels to blend across tile edges."}),
                "tile_calc": ("TILE_CALC", {"tooltip": "Tile layout metadata returned by DynamicTileSplit."}),
            },
            "optional": {
                "auto_save": ("BOOLEAN", {"default": False, "tooltip": "Automatically save the merged result to the output directory."}),
                "filename_prefix": ("STRING", {"default": "tile_merge", "tooltip": "Filename prefix used when auto-saving the merged image."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def process(self, images, blend, tile_calc, auto_save=False, filename_prefix="tile_merge"):
        filename_prefix = self._sanitize_prefix(filename_prefix)
        tile_height = images.shape[1]
        tile_width = images.shape[2]
        channels = images.shape[3]

        if isinstance(tile_calc, dict):
            overlap = tile_calc.get("overlap", 128)
            base_height = tile_calc.get("image_height", tile_height)
            base_width = tile_calc.get("image_width", tile_width)
            offset = tile_calc.get("offset", 0)
            base_tile_height = tile_calc.get("tile_height", tile_height)
            base_tile_width = tile_calc.get("tile_width", tile_width)
            blend_mode = tile_calc.get("blend_mode", "linear")
            tile_positions = tile_calc.get("tile_positions", None)
            rows = tile_calc.get("rows", 1)
            cols = tile_calc.get("cols", 1)
        elif isinstance(tile_calc, tuple):
            if len(tile_calc) >= 6:
                overlap, base_height, base_width, offset, base_tile_height, base_tile_width = tile_calc[:6]
            else:
                overlap, base_height, base_width, offset = tile_calc
                base_tile_height = tile_height
                base_tile_width = tile_width
            blend_mode = "linear"
            tile_positions = None
            rows = 1
            cols = 1
        else:
            raise ValueError(f"Unexpected tile_calc type: {type(tile_calc)}")

        scale_y = tile_height / base_tile_height if base_tile_height else 1.0
        scale_x = tile_width / base_tile_width if base_tile_width else 1.0
        average_scale = (scale_x + scale_y) / 2.0

        scaled_final_height = max(tile_height, int(round(base_height * scale_y)))
        scaled_final_width = max(tile_width, int(round(base_width * scale_x)))
        scaled_overlap = max(0, int(round(overlap * average_scale)))
        scaled_offset = int(round(offset * average_scale))

        tile_coordinates = generate_tiles(
            scaled_final_width, scaled_final_height, tile_width, tile_height, scaled_overlap, scaled_offset
        )

        canvas = torch.zeros(
            (1, scaled_final_height, scaled_final_width, channels), dtype=images.dtype, device=images.device
        )

        try:
            blender = get_blender(blend_mode, blend)
            use_advanced_blending = True
        except Exception as e:
            print(f"Warning: Could not initialize {blend_mode} blender ({e}), falling back to legacy blending")
            use_advanced_blending = False

        index = 0
        for tile_coordinate in tile_coordinates:
            if index >= len(images):
                print(f"Warning: More coordinates ({len(tile_coordinates)}) than tiles ({len(images)}). Stopping at tile {index}.")
                break

            image_tile = images[index:index+1]
            x = tile_coordinate[0]
            y = tile_coordinate[1]

            if use_advanced_blending and tile_positions and index < len(tile_positions):
                position = tile_positions[index]

                position_scaled = {
                    "index": position["index"],
                    "row": position["row"],
                    "col": position["col"],
                    "place_x": int(position["place_x"] * scale_x),
                    "place_y": int(position["place_y"] * scale_y),
                }

                tile_calc_for_blender = {
                    "image_height": scaled_final_height,
                    "image_width": scaled_final_width,
                    "overlap": scaled_overlap,
                    "rows": rows,
                    "cols": cols,
                }

                canvas = blender.blend_tiles(canvas, image_tile, position_scaled, tile_calc_for_blender)
            else:
                weight_matrix = torch.ones((tile_height, tile_width, channels), device=images.device)

                for i in range(blend):
                    weight = float(i) / blend
                    weight_matrix[i, :, :] *= weight
                    weight_matrix[-(i + 1), :, :] *= weight
                    weight_matrix[:, i, :] *= weight
                    weight_matrix[:, -(i + 1), :] *= weight

                old_tile = canvas[:, y:y+tile_height, x:x+tile_width, :]
                old_tile_count = (old_tile.abs().sum(dim=-1, keepdim=True) > 0).float()

                weight_matrix = (
                    weight_matrix * (old_tile_count != 0).float()
                    + (old_tile_count == 0).float()
                )

                blended = image_tile[0] * weight_matrix + old_tile[0] * (1 - weight_matrix)
                canvas[:, y:y+tile_height, x:x+tile_width, :] = blended.unsqueeze(0)

            index += 1

        saved_path = None
        if auto_save:
            try:
                saved_path = self._save_merged_image(canvas, filename_prefix)
            except Exception as exc:
                print(f"SimpleTilesUprezDynamicTileMerge: Failed to save merged image ({exc})")
        if saved_path:
            print(f"SimpleTilesUprezDynamicTileMerge: Saved merged image to {saved_path}")

        return [canvas]

    @staticmethod
    def _sanitize_prefix(name: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
        return sanitized or "tile_merge"

    def _save_merged_image(self, tensor: torch.Tensor, prefix: str) -> str:
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        full_path = os.path.join(output_dir, filename)

        image_tensor = tensor[0].clamp(0, 1)
        image_array = image_tensor.mul(255.0).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(image_array).save(full_path)
        return full_path


NODE_CLASS_MAPPINGS = {
    "DynamicTileSplit": DynamicTileSplit,
    "DynamicTileMerge": DynamicTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicTileSplit": "DynamicTileSplit",
    "DynamicTileMerge": "DynamicTileMerge",
}

