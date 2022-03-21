from time import perf_counter
from contextlib import contextmanager

@contextmanager
def time_this(name,use_timer=True):
    if use_timer:
        start = perf_counter()
        yield
        print('Time for', name, 'is', perf_counter() - start)
    else:
        yield
