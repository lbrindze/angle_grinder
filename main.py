import io
import mercantile
import numpy as np
import png
import xarray as xr

from fastapi import FastAPI
from PIL import Image
from fastapi.responses import StreamingResponse
from zarr import RedisStore

app = FastAPI()

store = RedisStore(prefix="zarr_test0", host="127.0.0.1", port=6379)


def resize(data_tile: xr.DataArray) -> np.ndarray:
    return Image.fromarray(data_tile.values).resize((512, 512))


def normalize(data_tile: xr.DataArray) -> np.ndarray:
    diff_from_min = data_tile - data_tile.min().values
    data_tile_range = data_tile.max().values - data_tile.min().values
    return np.floor(255 * diff_from_min / data_tile_range).astype(np.uint8)


def encode_as_png(vals: np.ndarray) -> io.BytesIO:
    zeros = np.zeros_like(vals)
    pixels = (vals, zeros, zeros)
    rgb = np.dstack(pixels).reshape(512, 512 * len(pixels))

    # write to buffer then flush out
    img = png.from_array(rgb, mode="RGB")
    buf = io.BytesIO()
    img.write(buf)
    buf.seek(0)
    return buf


@app.get("/{var}/{z}/{x}/{y}.png")
async def get_tile(var: str, z: int, x: int, y: int):
    tile_meta = mercantile.Tile(x=x, y=y, z=z)
    min_lon, min_lat, max_lon, max_lat = mercantile.bounds(tile_meta)

    # possible i/o bound could asyncify with threadpool
    data_tile = xr.open_zarr(store=store)[var].sel(
        longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)
    )

    normalized_tile = resize(normalize(data_tile))

    # rgb encode
    buf = encode_as_png(normalized_tile)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/{var}/{z}/{x}/{y}.json")
async def get_tile_meta(var: str, z: int, x: int, y: int):
    # Serves some metadata about tile...
    tile_meta = mercantile.Tile(x=x, y=y, z=z)
    min_lon, min_lat, max_lon, max_lat = mercantile.bounds(tile_meta)

    data_tile = xr.open_zarr(store=store)[var].sel(
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
