import mercantile
import os
import xarray as xr

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from functools import partial
from zarr import RedisStore

from image_utils import encode_as_png, resize, normalize


app = FastAPI()

store = partial(
    RedisStore,
    host=os.environ.get("REDIS_HOST", "127.0.0.1"),
    port=os.environ.get("REDIS_PORT", 6379),
)


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
