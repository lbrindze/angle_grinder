# angle_grinder

this project is in ALPHA, beware!

## Quick Start

Spin up redis locally, you could do:
```
$ docker run --rm -n redis -d -p 6739:6379 redis
```

Install deps
```
$ python -m venv ./env_angle_grinder
$ source env_angle_grinder/bin/activate
$ pip install -r requirements.txt
```

Load your dataset into a zarr store using the api directly!  Configure redis host and port using envvars `REDIS_HOST` and `REDIS_PORT`
to load data run the following replacing the contrived names with actual names you want
`curl -X POST -F netcdf_file=@PATH_TO_FILE.nc  http://localhost:8000/TEST_DATA_NAME/load_netcdf`

hit the endpoints with your web browser or curl!
e.g.

go to `http://localhost:8000/docs` for full api reference documentation.
go to `http://localhost:8000/my_test_data/variables` to get a list of variables
go to `http://localhost:8000/my_test_data/var_name/4/3/1.png` for a z,x,y tile on your data and var... or visit `http://localhost:8000/my_test_data/var_name/4/3/1.json` for corresponding metadata
to colorize add a query param `my_test_data/var_name/4/3/1.png?colorize=COLOR_MAP_NAME` per the available color schemes in "colormap.py"


