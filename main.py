import io
import mercantile
import numpy as np
import os
import png
import xarray as xr

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from functools import partial
from PIL import Image
from typing import Iterable, Callable
from zarr import RedisStore

app = FastAPI()

store = partial(
    RedisStore,
    host=os.environ.get("REDIS_HOST", "127.0.0.1"),
    port=os.environ.get("REDIS_PORT", 6379),
)


COLORMAP = [
    (0x8B, 0x49, 0xBD),
    (0x62, 0x4F, 0xAD),
    (0x41, 0x5B, 0xA0),
    (0x44, 0x77, 0xAA),
    (0x4C, 0x97, 0xB7),
    (0x59, 0xBC, 0xC8),
    (0x68, 0xE0, 0xD6),
    (0x67, 0xD0, 0xD0),
    (0x69, 0xB8, 0x5E),
    (0x8C, 0xB2, 0x3D),
    (0xE6, 0xCA, 0x44),
    (0xEC, 0xB1, 0x3F),
    (0xDD, 0xB8, 0x37),
    (0xE0, 0x4D, 0x2A),
    (0xC3, 0x36, 0x22),
    (0x95, 0x27, 0x17),
]


def resize(data_tile: xr.DataArray) -> np.ndarray:
    img = Image.fromarray(data_tile.values).resize((512, 512))
    return np.array(img)


def normalize(data_tile: xr.DataArray) -> np.ndarray:
    diff_from_min = data_tile - data_tile.min().values
    data_tile_range = data_tile.max().values - data_tile.min().values
    return np.floor(255 * diff_from_min / data_tile_range).astype(np.uint8)


def create_color_gradient(from_color, to_color, steps) -> np.ndarray:
    c = np.linspace(0, 1, steps)[:, None]
    x = np.ones((steps, 3))
    x[:, 0:3] = from_color
    y = np.ones((steps, 3))
    y[:, 0:3] = to_color
    gradient = x + (y - x) * c
    return gradient.astype(np.uint8)


def apply_colormap(colormap: Iterable[int]) -> Callable[[int], int]:
    num_colors = len(colormap)
    vals_per_tier = 255 // (num_colors - 1)
    gradients = np.concatenate(
        [
            create_color_gradient(colormap[i - 1], color, vals_per_tier)
            for i, color in enumerate(colormap)
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
            canvas[idx : idx + 3] = color_val
        return canvas.reshape(512, 512, 3)

    return mapper


def encode_as_png(vals: np.ndarray, colorize=False) -> io.BytesIO:
    if colorize:
        pixels = apply_colormap(COLORMAP)(vals)
        img = png.from_array(
            pixels.reshape(512, 512 * 3).astype(np.uint8), mode="RGB"
        )
    else:
        img = png.from_array(vals.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.write(buf)
    buf.seek(0)
    return buf


@app.get("/{prefix}/{var}/{z}/{x}/{y}.png")
async def get_tile(
    prefix: str, var: str, z: int, x: int, y: int, colorize: bool = False
):
    tile_meta = mercantile.Tile(x=x, y=y, z=z)
    min_lon, min_lat, max_lon, max_lat = mercantile.bounds(tile_meta)

    # possible i/o bound could asyncify with threadpool
    data_tile = xr.open_zarr(store=store(prefix=prefix))[var].sel(
        longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)
    )

    normalized_tile = resize(normalize(data_tile))

    # rgb encode
    buf = encode_as_png(normalized_tile, colorize=colorize)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/{prefix}/{var}/{z}/{x}/{y}.json")
async def get_tile_meta(prefix: str, var: str, z: int, x: int, y: int):
    # Serves some metadata about tile...
    tile_meta = mercantile.Tile(x=x, y=y, z=z)
    min_lon, min_lat, max_lon, max_lat = mercantile.bounds(tile_meta)

    data_tile = xr.open_zarr(store=store(prefix=prefix))[var].sel(
        longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)
    )
    print()
    print(data_tile.min().compute().values)

    return {
        "variable": var,
        "z": z,
        "x": x,
        "y": y,
        "lats": [min_lat, max_lat],
        "lons": [min_lon, max_lon],
        f"{var}Max": float(data_tile.max().compute()),
        f"{var}Min": float(data_tile.min().compute()),
        "units": data_tile.units,
        "description": data_tile.long_name,
    }
