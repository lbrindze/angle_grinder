import io
import numpy as np
import xarray as xr
import png
from PIL import Image

from typing import Iterable, Callable

import colormap


def resize(data_tile: xr.DataArray) -> np.ndarray:
    img = Image.fromarray(data_tile.values).resize((512, 512))
    return np.array(img)


def normalize(data_tile: xr.DataArray) -> np.ndarray:
    diff_from_min = data_tile - data_tile.min().values
    data_tile_range = data_tile.max().values - data_tile.min().values
    return np.floor(255 * diff_from_min / data_tile_range).astype(np.uint8)


def apply_colormap(color_map: Iterable[int]) -> Callable[[int], int]:
    num_colors = len(color_map)
    vals_per_tier = 255 // (num_colors - 1)
    gradients = np.concatenate(
        [
            create_color_gradient(color_map[i - 1], color, vals_per_tier)
            for i, color in enumerate(color_map)
            if i > 0
        ]
    )

    def mapper(vals):
        canvas = np.ones(512 * 512 * 3)
        for i, val in enumerate(vals.reshape(512 * 512)):
            try:
                color_val = gradients[val]
            except IndexError:
                color_val = gradients[-1]

            idx = 3 * i
            stride = idx + 3
            canvas[idx:stride] = color_val
        return canvas.reshape(512, 512, 3)

    return mapper


def encode_as_png(vals: np.ndarray, colorize=None) -> io.BytesIO:
    if colorize is None or colorize is "":
        pixels = vals.astype(np.uint8)
        mode = "L"
    else:
        try:
            mapping = getattr(colormap, colorize)
            pixels = (
                apply_colormap(mapping)(vals)
                .reshape(512, 512 * 3)
                .astype(np.uint8)
            )
            mode = "RGB"
        except AttributeError:
            print(f"No colormap with name {colorize} exists")
            pixels = vals.astype(np.uint8)
            mode = "L"

    img = png.from_array(pixels, mode=mode)
    buf = io.BytesIO()
    img.write(buf)
    buf.seek(0)
    return buf


def create_color_gradient(from_color, to_color, steps) -> np.ndarray:
    c = np.linspace(0, 1, steps)[:, None]
    x = np.ones((steps, 3))
    x[:, 0:3] = from_color
    y = np.ones((steps, 3))
    y[:, 0:3] = to_color
    gradient = x + (y - x) * c
    return gradient.astype(np.uint8)
