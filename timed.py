import time

def timed(func):
    """Decorator to measure execution time of allocation methods."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        result["time_taken_sec"] = end - start
        return result
    return wrapper