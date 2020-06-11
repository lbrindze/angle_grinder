import asyncio
import concurrent.futures
from functools import wraps


def run_async_threaded(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor()
        args = (*args, *[vals for _, vals in kwargs.items()])
        return await loop.run_in_executor(executor, func, *args)

    return wrapper


# as an alias...
run_async = run_async_threaded
