# angle_grinder

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

Load your dataset into a zarr store, and configure it to point to that prefix e.g. line 15 in main.py 

hit the endpoints with your web browser or curl!
e.g. for ecmwf try
`http://localhost:8000/t2m/3/1/1.png`
