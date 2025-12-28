import os
import re
from datetime import datetime

import torch
from PIL import Image

import folder_paths

from .blending import get_blender, get_available_blend_modes, AccumulationTileMerger


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
        blend_modes = get_available_blend_modes()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image tensor to split into tiles."}),
                "tile_width": ("INT", {"default": 512, "min": 1, "max": 10000, "tooltip": "Tile width in pixels before upscaling."}),
                "tile_height": ("INT", {"default": 512, "min": 1, "max": 10000, "tooltip": "Tile height in pixels before upscaling."}),
                "overlap": ("INT", {"default": 128, "min": 1, "max": 10000, "tooltip": "Overlap between adjacent tiles in pixels."}),
                "blend_mode": (blend_modes, {"default": "linear", "tooltip": "Blending algorithm for tile merging."}),
                "offset": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Initial offset applied to the first tile row and column."}),
            },
            "optional": {
                "feather_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 1.0,
                    "tooltip": "Feather as percentage of tile size (0=use overlap, 1-50=percentage). Overrides blend width calculation."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_CALC")
    FUNCTION = "process"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def process(self, image, tile_width, tile_height, overlap, blend_mode, offset, feather_percent=0.0):
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

            # Edge detection for edge-fallback strategy
            is_left_edge = (x_start == 0)
            is_right_edge = (x_end >= image_width)
            is_top_edge = (y_start == 0)
            is_bottom_edge = (y_end >= image_height)

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
                    "is_left_edge": is_left_edge,
                    "is_right_edge": is_right_edge,
                    "is_top_edge": is_top_edge,
                    "is_bottom_edge": is_bottom_edge,
                }
            )

        tiles_tensor = torch.stack(image_tiles).squeeze(1)

        # Calculate blend width from feather percentage if specified
        if feather_percent > 0:
            avg_tile_size = (tile_width + tile_height) / 2
            calculated_blend = int(avg_tile_size * feather_percent / 100.0)
        else:
            calculated_blend = overlap  # Default to overlap

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
            "feather_percent": feather_percent,
            "calculated_blend": calculated_blend,
        }

        return (tiles_tensor, tile_calc)


class DynamicTileMerge:
    @classmethod
    def INPUT_TYPES(s):
        blend_modes = get_available_blend_modes()
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of tiles to merge back into the full image."}),
                "blend": ("INT", {"default": 64, "min": 0, "max": 4096, "tooltip": "Number of pixels to blend across tile edges (0=auto from tile_calc)."}),
                "tile_calc": ("TILE_CALC", {"tooltip": "Tile layout metadata returned by DynamicTileSplit."}),
            },
            "optional": {
                "processing_mode": (["batch", "sequential"], {
                    "default": "batch",
                    "tooltip": "batch=all tiles in memory (fast), sequential=one at a time (low VRAM)"
                }),
                "use_accumulation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use accumulation buffer blending for order-independent results"
                }),
                "edge_fallback": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Adjust blend weights at image boundaries to prevent artifacts"
                }),
                "auto_save": ("BOOLEAN", {"default": False, "tooltip": "Automatically save the merged result to the output directory."}),
                "filename_prefix": ("STRING", {"default": "tile_merge", "tooltip": "Filename prefix used when auto-saving the merged image."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def process(
        self,
        images,
        blend,
        tile_calc,
        processing_mode="batch",
        use_accumulation=False,
        edge_fallback=True,
        auto_save=False,
        filename_prefix="tile_merge"
    ):
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
            feather_percent = tile_calc.get("feather_percent", 0.0)
            calculated_blend = tile_calc.get("calculated_blend", None)
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
            feather_percent = 0.0
            calculated_blend = None
        else:
            raise ValueError(f"Unexpected tile_calc type: {type(tile_calc)}")

        scale_y = tile_height / base_tile_height if base_tile_height else 1.0
        scale_x = tile_width / base_tile_width if base_tile_width else 1.0
        average_scale = (scale_x + scale_y) / 2.0

        scaled_final_height = max(tile_height, int(round(base_height * scale_y)))
        scaled_final_width = max(tile_width, int(round(base_width * scale_x)))
        scaled_overlap = max(0, int(round(overlap * average_scale)))
        scaled_offset = int(round(offset * average_scale))

        # Determine blend width: parameter > calculated from feather > scaled overlap
        if blend > 0:
            effective_blend = int(blend * average_scale)
        elif calculated_blend is not None and calculated_blend > 0:
            effective_blend = int(calculated_blend * average_scale)
        else:
            effective_blend = scaled_overlap

        tile_coordinates = generate_tiles(
            scaled_final_width, scaled_final_height, tile_width, tile_height, scaled_overlap, scaled_offset
        )

        # Use accumulation buffer blending if requested
        if use_accumulation:
            canvas = self._merge_with_accumulation(
                images, tile_coordinates, tile_positions, tile_calc,
                scaled_final_height, scaled_final_width, channels,
                effective_blend, blend_mode, scale_x, scale_y, rows, cols, edge_fallback
            )
        elif processing_mode == "sequential":
            canvas = self._merge_sequential(
                images, tile_coordinates, tile_positions, tile_calc,
                scaled_final_height, scaled_final_width, channels,
                effective_blend, blend_mode, scale_x, scale_y, rows, cols, edge_fallback
            )
        else:
            canvas = self._merge_batch(
                images, tile_coordinates, tile_positions, tile_calc,
                scaled_final_height, scaled_final_width, channels,
                effective_blend, blend_mode, scale_x, scale_y, rows, cols, edge_fallback
            )

        saved_path = None
        if auto_save:
            try:
                saved_path = self._save_merged_image(canvas, filename_prefix)
            except Exception as exc:
                print(f"SimpleTilesUprezDynamicTileMerge: Failed to save merged image ({exc})")
        if saved_path:
            print(f"SimpleTilesUprezDynamicTileMerge: Saved merged image to {saved_path}")

        return [canvas]

    def _merge_with_accumulation(
        self, images, tile_coordinates, tile_positions, tile_calc,
        height, width, channels, blend, blend_mode, scale_x, scale_y, rows, cols, edge_fallback
    ):
        """Merge using accumulation buffer for order-independent blending."""
        # Extract weight mode from blend_mode
        weight_mode = "gaussian" if blend_mode in ["gaussian", "accumulation"] else \
                      "cosine" if blend_mode == "cosine" else "linear"

        merger = AccumulationTileMerger(
            height=height,
            width=width,
            channels=channels,
            blend_width=blend,
            weight_mode=weight_mode,
            edge_fallback=edge_fallback,
            device=images.device,
            dtype=images.dtype
        )

        tile_calc_for_merger = {
            "image_height": height,
            "image_width": width,
            "overlap": blend,
            "rows": rows,
            "cols": cols,
        }

        for index, tile_coordinate in enumerate(tile_coordinates):
            if index >= len(images):
                break

            image_tile = images[index:index+1]
            x = tile_coordinate[0]
            y = tile_coordinate[1]

            if tile_positions and index < len(tile_positions):
                position = tile_positions[index]
                position_scaled = {
                    "index": position["index"],
                    "row": position["row"],
                    "col": position["col"],
                    "place_x": int(position["place_x"] * scale_x),
                    "place_y": int(position["place_y"] * scale_y),
                }
            else:
                # Fallback position calculation
                row = y // max(1, (images.shape[1] - blend))
                col = x // max(1, (images.shape[2] - blend))
                position_scaled = {
                    "index": index,
                    "row": row,
                    "col": col,
                    "place_x": x,
                    "place_y": y,
                }

            merger.add_tile(image_tile, position_scaled, tile_calc_for_merger)

        return merger.finalize()

    def _merge_sequential(
        self, images, tile_coordinates, tile_positions, tile_calc,
        height, width, channels, blend, blend_mode, scale_x, scale_y, rows, cols, edge_fallback
    ):
        """Merge tiles one at a time to minimize VRAM usage."""
        canvas = torch.zeros(
            (1, height, width, channels), dtype=images.dtype, device=images.device
        )

        try:
            blender = get_blender(blend_mode, blend)
            use_advanced_blending = True
        except Exception as e:
            print(f"Warning: Could not initialize {blend_mode} blender ({e}), falling back to legacy blending")
            use_advanced_blending = False

        tile_calc_for_blender = {
            "image_height": height,
            "image_width": width,
            "overlap": blend,
            "rows": rows,
            "cols": cols,
        }

        for index, tile_coordinate in enumerate(tile_coordinates):
            if index >= len(images):
                print(f"Warning: More coordinates ({len(tile_coordinates)}) than tiles ({len(images)}). Stopping at tile {index}.")
                break

            # Load single tile to minimize memory
            image_tile = images[index:index+1]
            x = tile_coordinate[0]
            y = tile_coordinate[1]
            tile_height = image_tile.shape[1]
            tile_width = image_tile.shape[2]

            if use_advanced_blending and tile_positions and index < len(tile_positions):
                position = tile_positions[index]
                position_scaled = {
                    "index": position["index"],
                    "row": position["row"],
                    "col": position["col"],
                    "place_x": int(position["place_x"] * scale_x),
                    "place_y": int(position["place_y"] * scale_y),
                }
                canvas = blender.blend_tiles(canvas, image_tile, position_scaled, tile_calc_for_blender)
            else:
                canvas = self._legacy_blend_tile(canvas, image_tile, x, y, blend, tile_height, tile_width, channels)

        return canvas

    def _merge_batch(
        self, images, tile_coordinates, tile_positions, tile_calc,
        height, width, channels, blend, blend_mode, scale_x, scale_y, rows, cols, edge_fallback
    ):
        """Standard batch merge - all tiles in memory."""
        canvas = torch.zeros(
            (1, height, width, channels), dtype=images.dtype, device=images.device
        )

        try:
            blender = get_blender(blend_mode, blend)
            use_advanced_blending = True
        except Exception as e:
            print(f"Warning: Could not initialize {blend_mode} blender ({e}), falling back to legacy blending")
            use_advanced_blending = False

        tile_calc_for_blender = {
            "image_height": height,
            "image_width": width,
            "overlap": blend,
            "rows": rows,
            "cols": cols,
        }

        index = 0
        for tile_coordinate in tile_coordinates:
            if index >= len(images):
                print(f"Warning: More coordinates ({len(tile_coordinates)}) than tiles ({len(images)}). Stopping at tile {index}.")
                break

            image_tile = images[index:index+1]
            x = tile_coordinate[0]
            y = tile_coordinate[1]
            tile_height = image_tile.shape[1]
            tile_width = image_tile.shape[2]

            if use_advanced_blending and tile_positions and index < len(tile_positions):
                position = tile_positions[index]
                position_scaled = {
                    "index": position["index"],
                    "row": position["row"],
                    "col": position["col"],
                    "place_x": int(position["place_x"] * scale_x),
                    "place_y": int(position["place_y"] * scale_y),
                }
                canvas = blender.blend_tiles(canvas, image_tile, position_scaled, tile_calc_for_blender)
            else:
                canvas = self._legacy_blend_tile(canvas, image_tile, x, y, blend, tile_height, tile_width, channels)

            index += 1

        return canvas

    def _legacy_blend_tile(self, canvas, image_tile, x, y, blend, tile_height, tile_width, channels):
        """Legacy blending fallback."""
        weight_matrix = torch.ones((tile_height, tile_width, channels), device=canvas.device)

        for i in range(blend):
            weight = float(i) / blend if blend > 0 else 1.0
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

        return canvas

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


class OptimalTileSizeCalculator:
    """
    Calculates optimal tile dimensions based on image size, overlap requirements,
    and optional VRAM budget constraints.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image to calculate optimal tile size for."}),
                "target_tile_count": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Target number of tiles (will adjust to fit evenly)"
                }),
                "min_overlap_percent": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 1.0,
                    "tooltip": "Minimum overlap as percentage of tile size"
                }),
                "divisible_by": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Tile dimensions must be divisible by this (8 for latent space compatibility)"
                }),
            },
            "optional": {
                "max_tile_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16777216,
                    "tooltip": "Maximum pixels per tile (0=no limit). Use for VRAM constraints. 512x512=262144, 1024x1024=1048576"
                }),
                "prefer_square": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prefer square tiles over matching aspect ratio"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("tile_width", "tile_height", "overlap", "tile_count", "info")
    FUNCTION = "calculate"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def calculate(
        self,
        image,
        target_tile_count,
        min_overlap_percent,
        divisible_by,
        max_tile_pixels=0,
        prefer_square=True
    ):
        img_height = image.shape[1]
        img_width = image.shape[2]

        # Calculate grid dimensions
        import math
        if prefer_square:
            # Try to make grid as square as possible
            grid_cols = int(math.ceil(math.sqrt(target_tile_count * img_width / img_height)))
            grid_rows = int(math.ceil(target_tile_count / grid_cols))
        else:
            # Match image aspect ratio
            aspect = img_width / img_height
            grid_cols = int(math.ceil(math.sqrt(target_tile_count * aspect)))
            grid_rows = int(math.ceil(target_tile_count / grid_cols))

        grid_cols = max(1, grid_cols)
        grid_rows = max(1, grid_rows)
        actual_tile_count = grid_rows * grid_cols

        # Calculate base tile size without overlap
        base_tile_width = img_width / grid_cols
        base_tile_height = img_height / grid_rows

        # Add overlap
        overlap_factor = 1.0 + (min_overlap_percent / 100.0)
        tile_width = int(base_tile_width * overlap_factor)
        tile_height = int(base_tile_height * overlap_factor)

        # Round to divisibility requirement
        tile_width = max(divisible_by, (tile_width // divisible_by) * divisible_by)
        tile_height = max(divisible_by, (tile_height // divisible_by) * divisible_by)

        # Apply max pixels constraint
        if max_tile_pixels > 0:
            current_pixels = tile_width * tile_height
            if current_pixels > max_tile_pixels:
                scale = math.sqrt(max_tile_pixels / current_pixels)
                tile_width = max(divisible_by, int((tile_width * scale) // divisible_by) * divisible_by)
                tile_height = max(divisible_by, int((tile_height * scale) // divisible_by) * divisible_by)

        # Calculate actual overlap in pixels
        overlap_x = max(0, tile_width - int(img_width / grid_cols))
        overlap_y = max(0, tile_height - int(img_height / grid_rows))
        overlap = min(overlap_x, overlap_y)

        # Ensure overlap doesn't exceed tile size
        overlap = min(overlap, tile_width // 2, tile_height // 2)

        # Build info string
        info_lines = [
            f"Image: {img_width}x{img_height}",
            f"Grid: {grid_cols}x{grid_rows} = {actual_tile_count} tiles",
            f"Tile size: {tile_width}x{tile_height}",
            f"Overlap: {overlap}px ({overlap*100/min(tile_width, tile_height):.1f}%)",
            f"Pixels per tile: {tile_width * tile_height:,}",
        ]
        info = "\n".join(info_lines)

        return (tile_width, tile_height, overlap, actual_tile_count, info)


class TileBoundaryPreview:
    """
    Visualizes tile boundaries overlaid on the source image.
    Useful for debugging and tuning tile size/overlap before processing.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image to preview tile boundaries on."}),
                "tile_width": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "tile_height": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 5000}),
            },
            "optional": {
                "offset": ("INT", {"default": 0, "min": 0, "max": 5000}),
                "line_color": (["red", "green", "blue", "yellow", "white", "black"], {"default": "red"}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10}),
                "show_overlap": ("BOOLEAN", {"default": True, "tooltip": "Highlight overlap regions"}),
                "overlap_opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "info")
    FUNCTION = "preview"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def preview(
        self,
        image,
        tile_width,
        tile_height,
        overlap,
        offset=0,
        line_color="red",
        line_width=2,
        show_overlap=True,
        overlap_opacity=0.3
    ):
        import torch

        img_height = image.shape[1]
        img_width = image.shape[2]
        channels = image.shape[3]

        # Generate tile coordinates
        tile_coords = generate_tiles(img_width, img_height, tile_width, tile_height, overlap, offset)

        # Create output canvas
        preview = image.clone()

        # Color mapping
        color_map = {
            "red": [1.0, 0.0, 0.0],
            "green": [0.0, 1.0, 0.0],
            "blue": [0.0, 0.0, 1.0],
            "yellow": [1.0, 1.0, 0.0],
            "white": [1.0, 1.0, 1.0],
            "black": [0.0, 0.0, 0.0],
        }
        color = color_map.get(line_color, [1.0, 0.0, 0.0])

        # Overlap highlight color (semi-transparent)
        overlap_color = torch.tensor([1.0, 0.5, 0.0], device=image.device)  # Orange

        # Draw overlap regions first (if enabled)
        if show_overlap and overlap > 0:
            overlap_mask = torch.zeros((img_height, img_width), device=image.device)

            for x, y in tile_coords:
                x_end = min(x + tile_width, img_width)
                y_end = min(y + tile_height, img_height)

                # Mark this tile's area
                tile_mask = torch.zeros_like(overlap_mask)
                tile_mask[y:y_end, x:x_end] = 1.0

                # Accumulate overlaps
                overlap_mask += tile_mask

            # Regions with overlap_mask > 1 are overlapping
            overlap_regions = (overlap_mask > 1).float()

            if overlap_regions.any():
                for c in range(min(channels, 3)):
                    preview[0, :, :, c] = preview[0, :, :, c] * (1 - overlap_regions * overlap_opacity) + \
                                          overlap_color[c] * overlap_regions * overlap_opacity

        # Draw tile boundaries
        for x, y in tile_coords:
            x_end = min(x + tile_width, img_width)
            y_end = min(y + tile_height, img_height)

            # Draw rectangle
            for lw in range(line_width):
                # Top edge
                if y + lw < img_height:
                    for c in range(min(channels, 3)):
                        preview[0, y + lw, x:x_end, c] = color[c]
                # Bottom edge
                if y_end - 1 - lw >= 0 and y_end - 1 - lw < img_height:
                    for c in range(min(channels, 3)):
                        preview[0, y_end - 1 - lw, x:x_end, c] = color[c]
                # Left edge
                if x + lw < img_width:
                    for c in range(min(channels, 3)):
                        preview[0, y:y_end, x + lw, c] = color[c]
                # Right edge
                if x_end - 1 - lw >= 0 and x_end - 1 - lw < img_width:
                    for c in range(min(channels, 3)):
                        preview[0, y:y_end, x_end - 1 - lw, c] = color[c]

        # Calculate grid info
        rows = 0
        cols = 0
        for x, y in tile_coords:
            row = y // (tile_height - overlap) if overlap < tile_height else 0
            col = x // (tile_width - overlap) if overlap < tile_width else 0
            rows = max(rows, row + 1)
            cols = max(cols, col + 1)

        info_lines = [
            f"Image: {img_width}x{img_height}",
            f"Tile size: {tile_width}x{tile_height}",
            f"Overlap: {overlap}px",
            f"Grid: {cols}x{rows}",
            f"Total tiles: {len(tile_coords)}",
        ]
        info = "\n".join(info_lines)

        return (preview, info)


class UpscaleAwareTileMerge:
    """
    Enhanced tile merger with automatic upscale detection and subpixel-accurate placement.
    Detects upscale factor from tile dimensions vs. original tile_calc metadata.
    """

    @classmethod
    def INPUT_TYPES(s):
        blend_modes = get_available_blend_modes()
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of (possibly upscaled) tiles to merge."}),
                "tile_calc": ("TILE_CALC", {"tooltip": "Original tile layout metadata from DynamicTileSplit."}),
            },
            "optional": {
                "blend_mode_override": (["auto"] + blend_modes, {
                    "default": "auto",
                    "tooltip": "Override blend mode (auto=use tile_calc setting)"
                }),
                "blend_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Override blend width (0=auto-calculate from upscale factor)"
                }),
                "use_accumulation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use accumulation buffer for order-independent blending"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("merged", "detected_scale", "info")
    FUNCTION = "merge"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def merge(
        self,
        images,
        tile_calc,
        blend_mode_override="auto",
        blend_override=0,
        use_accumulation=False
    ):
        tile_height = images.shape[1]
        tile_width = images.shape[2]
        channels = images.shape[3]

        # Extract original dimensions
        if isinstance(tile_calc, dict):
            base_tile_height = tile_calc.get("tile_height", tile_height)
            base_tile_width = tile_calc.get("tile_width", tile_width)
            base_height = tile_calc.get("image_height", tile_height)
            base_width = tile_calc.get("image_width", tile_width)
            overlap = tile_calc.get("overlap", 128)
            offset = tile_calc.get("offset", 0)
            blend_mode = tile_calc.get("blend_mode", "linear")
            tile_positions = tile_calc.get("tile_positions", None)
            rows = tile_calc.get("rows", 1)
            cols = tile_calc.get("cols", 1)
        else:
            raise ValueError("tile_calc must be a dictionary")

        # Detect upscale factor
        scale_y = tile_height / base_tile_height if base_tile_height > 0 else 1.0
        scale_x = tile_width / base_tile_width if base_tile_width > 0 else 1.0
        detected_scale = (scale_x + scale_y) / 2.0

        # Calculate scaled dimensions
        scaled_height = int(round(base_height * scale_y))
        scaled_width = int(round(base_width * scale_x))
        scaled_overlap = int(round(overlap * detected_scale))
        scaled_offset = int(round(offset * detected_scale))

        # Determine blend width
        if blend_override > 0:
            effective_blend = blend_override
        else:
            effective_blend = scaled_overlap

        # Determine blend mode
        if blend_mode_override != "auto":
            blend_mode = blend_mode_override

        # Generate scaled tile coordinates
        tile_coordinates = generate_tiles(
            scaled_width, scaled_height, tile_width, tile_height, scaled_overlap, scaled_offset
        )

        # Create canvas
        canvas = torch.zeros(
            (1, scaled_height, scaled_width, channels),
            dtype=images.dtype,
            device=images.device
        )

        tile_calc_for_blender = {
            "image_height": scaled_height,
            "image_width": scaled_width,
            "overlap": scaled_overlap,
            "rows": rows,
            "cols": cols,
        }

        if use_accumulation:
            # Use accumulation buffer
            weight_mode = "gaussian" if blend_mode in ["gaussian", "accumulation"] else \
                          "cosine" if blend_mode == "cosine" else "linear"

            merger = AccumulationTileMerger(
                height=scaled_height,
                width=scaled_width,
                channels=channels,
                blend_width=effective_blend,
                weight_mode=weight_mode,
                edge_fallback=True,
                device=images.device,
                dtype=images.dtype
            )

            for index, coord in enumerate(tile_coordinates):
                if index >= len(images):
                    break

                tile = images[index:index+1]
                x, y = coord

                if tile_positions and index < len(tile_positions):
                    pos = tile_positions[index]
                    position = {
                        "index": pos["index"],
                        "row": pos["row"],
                        "col": pos["col"],
                        "place_x": int(pos["place_x"] * scale_x),
                        "place_y": int(pos["place_y"] * scale_y),
                    }
                else:
                    position = {
                        "index": index,
                        "row": 0,
                        "col": 0,
                        "place_x": x,
                        "place_y": y,
                    }

                merger.add_tile(tile, position, tile_calc_for_blender)

            canvas = merger.finalize()
        else:
            # Standard blending
            try:
                blender = get_blender(blend_mode, effective_blend)
            except Exception as e:
                print(f"Warning: Could not initialize {blend_mode} blender ({e})")
                blender = None

            for index, coord in enumerate(tile_coordinates):
                if index >= len(images):
                    break

                tile = images[index:index+1]
                x, y = coord

                if blender and tile_positions and index < len(tile_positions):
                    pos = tile_positions[index]
                    position = {
                        "index": pos["index"],
                        "row": pos["row"],
                        "col": pos["col"],
                        "place_x": int(pos["place_x"] * scale_x),
                        "place_y": int(pos["place_y"] * scale_y),
                    }
                    canvas = blender.blend_tiles(canvas, tile, position, tile_calc_for_blender)
                else:
                    # Fallback simple placement
                    th, tw = tile.shape[1], tile.shape[2]
                    canvas[:, y:y+th, x:x+tw, :] = tile

        # Build info string
        info_lines = [
            f"Detected scale: {detected_scale:.2f}x",
            f"Original: {base_width}x{base_height}",
            f"Output: {scaled_width}x{scaled_height}",
            f"Tile size: {tile_width}x{tile_height}",
            f"Blend: {effective_blend}px ({blend_mode})",
        ]
        info = "\n".join(info_lines)

        return (canvas, detected_scale, info)


NODE_CLASS_MAPPINGS = {
    "DynamicTileSplit": DynamicTileSplit,
    "DynamicTileMerge": DynamicTileMerge,
    "OptimalTileSizeCalculator": OptimalTileSizeCalculator,
    "TileBoundaryPreview": TileBoundaryPreview,
    "UpscaleAwareTileMerge": UpscaleAwareTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicTileSplit": "Dynamic Tile Split",
    "DynamicTileMerge": "Dynamic Tile Merge",
    "OptimalTileSizeCalculator": "Optimal Tile Size Calculator",
    "TileBoundaryPreview": "Tile Boundary Preview",
    "UpscaleAwareTileMerge": "Upscale-Aware Tile Merge",
}
