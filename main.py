import mercantile
import xarray as xr
import netCDF4

from async_lru import alru_cache
from fastapi import FastAPI, File
from fastapi.responses import StreamingResponse

from image_utils import encode_as_png, resize, normalize
from data_utils import load_da, get_absolute_min_max, data_tile_meta, get_store


app = FastAPI()


@app.get("/healthz")
async def health_check():
    return {"status": "OK"}


@app.get("/{prefix}/variables")
async def get_variables(prefix: str):
    data_tile = xr.open_zarr(store=get_store(prefix))

    return {"variables": [var for var in data_tile.variables]}


@alru_cache(maxsize=32)
async def get_cached_tile_buffer(
    prefix: str, var: str, z: int, x: int, y: int, colorize: str = None
):
    print("calculate tile metadata")
    tile_meta = mercantile.Tile(x=x, y=y, z=z)
    print("getting bounds")
    min_lon, min_lat, max_lon, max_lat = mercantile.bounds(tile_meta)

    print("loading data array")
    da = await load_da(prefix, var)

    print("indexing tile selection")
    data_tile = da.sel(
        longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)
    )

    print("getting min/max")
    abs_min, abs_max = get_absolute_min_max(prefix, var)

    print("Slicing and dicing data")
    normalized_tile = resize(
        normalize(data_tile, max_val=abs_max, min_val=abs_min)
    )

    print("encoding png")
    return await encode_as_png(normalized_tile, colorize=colorize)


@app.get("/{prefix}/{var}/{z}/{x}/{y}.png")
async def get_tile(
    prefix: str, var: str, z: int, x: int, y: int, colorize: str = None
):
    print("Request Received")
    buf = await get_cached_tile_buffer(prefix, var, z, x, y, colorize=colorize)

    print("streaming response")
    return StreamingResponse(buf, media_type="image/png")


@app.get("/{prefix}/{var}/{z}/{x}/{y}.json")
async def get_tile_meta(prefix: str, var: str, z: int, x: int, y: int):
    # Serves some metadata about tile...
    return data_tile_meta(prefix, var, (z, x, y))


@app.post("/{prefix}/load_netcdf")
async def load_dataset(prefix: str, netcdf_file: bytes = File(...)):
    print("loading dataset into buffer...")
    nc4_ds = netCDF4.Dataset(f"{prefix}-zarr-prep", memory=netcdf_file)
    print("creating backend store...")
    mmap_file = xr.backends.NetCDF4DataStore(nc4_ds)
    print("opening dataset...")
    ds = xr.open_dataset(mmap_file)
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    if ds.latitude[0] < ds.latitude[-1]:
        # flip lat order
        ds = ds.sel(latitude=slice(None, None, -1))
    print("saving to redis...")
    ds.to_zarr(store=get_store(prefix), mode="w")

    return {
        "file_size": len(netcdf_file),
        "variables": [var for var in ds.variables],
    }
