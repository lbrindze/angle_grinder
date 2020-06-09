import mercantile
import numpy as np
import png
import tempfile
import xarray as xr

from fastapi import FastAPI
from PIL import Image
from starlette.responses import FileResponse
from zarr import RedisStore


app = FastAPI()

store = RedisStore(prefix="zarr_test0", host="127.0.0.1", port=6379)


@app.get("/{var}/{z}/{x}/{y}.png")
async def get_tile(var: str, z: int, x: int, y: int):
    tile_meta = mercantile.Tile(x=x, y=y, z=z)
    min_lon, min_lat, max_lon, max_lat = mercantile.bounds(tile_meta)
    print(min_lon, min_lat, max_lon, max_lat)

    # possible i/o bound could asyncify with threadpool
    data_tile = xr.open_zarr(store=store)[var].sel(
        longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)
    )

    # resize
    vals = Image.fromarray(data_tile.values).resize((512, 512))

    # normalize
    diff_from_min = vals - data_tile.min().values
    data_tile_range = data_tile.max().values - data_tile.min().values
    normalized_tile = np.floor(255 * diff_from_min / data_tile_range).astype(
        np.uint8
    )

    # rgb encode
    zeros = np.zeros_like(normalized_tile)
    pixels = (normalized_tile, zeros, zeros)
    rgb = np.dstack(pixels).reshape(512, 512 * len(pixels))

    # write to buffer then flush out
    img = png.from_array(rgb, mode="RGB")
    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".png", delete=False
    ) as f:
        img.write(f)
        return FileResponse(f.name, media_type="image/png")


@app.get("/{var}/{z}/{x}/{y}.json")
async def get_tile_meta(var: str, z: int, x: int, y: int):
    # Serves some metadata about tile...

    return {"variable": var, "z": z, "x": x, "y": y, "lats": [], "lons": []}
