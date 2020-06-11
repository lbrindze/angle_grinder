import mercantile
import os
from functools import lru_cache
import xarray as xr
from zarr import RedisStore

import redis

redis_pool = redis.ConnectionPool(
    host=os.environ.get("REDIS_HOST", "127.0.0.1"),
    port=os.environ.get("REDIS_PORT", 6379),
    db=os.environ.get("REDIS_DB", 0),
)

_stores = {}


def get_store(prefix):
    global _stores
    try:
        store = _stores[prefix]
    except KeyError:
        store = RedisStore(prefix=prefix, connection_pool=redis_pool)
        _stores[prefix] = store

    return store


@lru_cache(128)
def get_absolute_min_max(prefix, var):
    data_tile = xr.open_zarr(store=get_store(prefix))[var]
    return data_tile.min().compute(), data_tile.max().compute()


@lru_cache(256)
def data_tile_meta(prefix, var, tile_pos):
    z, x, y = tile_pos
    tile_meta = mercantile.Tile(x=x, y=y, z=z)
    min_lon, min_lat, max_lon, max_lat = mercantile.bounds(tile_meta)

    data_tile = xr.open_zarr(store=get_store(prefix))[var]
    data_tile = data_tile.sel(
        longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)
    )

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
