from time import perf_counter
from contextlib import contextmanager

@contextmanager
def time_this(name):
    start = perf_counter()
    yield
    print('Time for', name, 'is', perf_counter() - start)
