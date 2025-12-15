import torch


class TileSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image tensor to split into grid tiles."}),
                "tile_height": (
                    "INT",
                    {
                        "default": 64,
                        "min": 64,
                        "max": 4096,
                        "tooltip": "Tile height in pixels for the legacy splitter.",
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 64,
                        "min": 64,
                        "max": 4096,
                        "tooltip": "Tile width in pixels for the legacy splitter.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Horizontal overlap in pixels between legacy tiles.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "split"
    CATEGORY = "SimpleTiles Uprez/Legacy"

    def split(self, image, tile_height, tile_width, overlap):
        height, width = image.shape[1], image.shape[2]
        overlap_x = overlap
        overlap_y = int(overlap * (tile_height / tile_width))

        tiles = []
        step_y = max(1, tile_height - overlap_y)
        step_x = max(1, tile_width - overlap_x)

        for y in range(0, height - tile_height + 1, step_y):
            for x in range(0, width - tile_width + 1, step_x):
                tile = image[:, y : y + tile_height, x : x + tile_width, :]
                tiles.append(tile)

        if not tiles:
            return [image]

        tiles_tensor = torch.stack(tiles).squeeze(1)
        return [tiles_tensor]


class TileMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Tiles to blend back into a single image."}),
                "overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Overlap value that was used when splitting the tiles.",
                    },
                ),
                "blend": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Blend width in pixels to feather tile seams.",
                    },
                ),
                "final_height": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 9 * 4096,
                        "tooltip": "Target height of the reconstructed image.",
                    },
                ),
                "final_width": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 9 * 4096,
                        "tooltip": "Target width of the reconstructed image.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_tiles"
    CATEGORY = "SimpleTiles Uprez/Legacy"

    def blend_tiles(self, images, overlap, blend, final_height, final_width):
        tiles = images
        tile_height, tile_width = images.shape[1], images.shape[2]
        original_shape = (1, final_height, final_width, 3)
        overlap_x = overlap
        overlap_y = int(overlap * (tile_height / tile_width))

        batch, height, width, channels = original_shape
        output = torch.zeros(original_shape, dtype=tiles.dtype)
        count = torch.zeros(original_shape, dtype=tiles.dtype)

        step_y = max(1, tile_height - overlap_y)
        step_x = max(1, tile_width - overlap_x)
        rows = (height - tile_height) // step_y + 1
        cols = (width - tile_width) // step_x + 1

        center_row, center_col = rows // 2, cols // 2

        distances = []
        for i in range(rows):
            for j in range(cols):
                distance = abs(i - center_row) + abs(j - center_col)
                distances.append(distance)

        reverse_custom_order = sorted(range(len(distances)), key=lambda k: distances[k])
        custom_order = reverse_custom_order[::-1]

        ys = [y for y in range(0, height - tile_height + 1, step_y)]
        xs = [x for x in range(0, width - tile_width + 1, step_x)]
        for idx in custom_order:
            y = ys[idx // len(xs)]
            x = xs[idx % len(xs)]

            tile = tiles[idx]

            weight_matrix = torch.ones((tile_height, tile_width, channels))
            for i in range(blend):
                weight = float(i) / blend
                weight_matrix[i, :, :] *= weight
                weight_matrix[-(i + 1), :, :] *= weight
                weight_matrix[:, i, :] *= weight
                weight_matrix[:, -(i + 1), :] *= weight

            old_tile = output[:, y : y + tile_height, x : x + tile_width, :]
            old_tile_count = count[:, y : y + tile_height, x : x + tile_width, :]

            weight_matrix = (
                weight_matrix * (old_tile_count != 0).float()
                + (old_tile_count == 0).float()
            )

            tile = tile * weight_matrix + old_tile * (1 - weight_matrix)

            output[:, y : y + tile_height, x : x + tile_width, :] = tile
            count[:, y : y + tile_height, x : x + tile_width, :] = 1

        return [output]


class TileCalc:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tile_height": (
                    "INT",
                    {
                        "default": 64,
                        "min": 64,
                        "max": 4096,
                        "tooltip": "Height of each tile in pixels.",
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 64,
                        "min": 64,
                        "max": 4096,
                        "tooltip": "Width of each tile in pixels.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Overlap between tiles in pixels.",
                    },
                ),
                "tile_width_n": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 9,
                        "tooltip": "Number of tiles across the width.",
                    },
                ),
                "tile_height_n": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 9,
                        "tooltip": "Number of tiles across the height.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("final_height", "final_width")
    FUNCTION = "calc"
    CATEGORY = "SimpleTiles Uprez/Legacy"

    def calc(self, tile_height, tile_width, overlap, tile_width_n, tile_height_n):
        overlap_x = overlap
        overlap_y = int(overlap * (tile_height / tile_width))

        final_height = tile_height * tile_height_n - overlap_y * (tile_height_n - 1)
        final_width = tile_width * tile_width_n - overlap_x * (tile_width_n - 1)
        print("Final height: {}, Final width: {}".format(final_height, final_width))

        return [final_height, final_width]


NODE_CLASS_MAPPINGS = {
    "TileSplit": TileSplit,
    "TileMerge": TileMerge,
    "TileCalc": TileCalc,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileSplit": "TileSplit",
    "TileMerge": "TileMerge",
    "TileCalc": "TileCalc",
}

