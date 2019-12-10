import contextlib
import time

import numpy

import cupy


class _PerfCaseResult(object):
    def __init__(self, name, ts):
        assert ts.ndim == 2 and ts.shape[0] == 2 and ts.shape[1] > 0
        self.name = name
        self._ts = ts

    @property
    def cpu_times(self):
        return self._ts[0]

    @property
    def gpu_times(self):
        return self._ts[1]

    @staticmethod
    def _to_str_per_item(t):
        assert t.size > 0
        t *= 1e6

        s = ' {:9.03f} us'.format(t.mean())
        if t.size > 1:
            s += '   +/-{:6.03f} (min:{:9.03f} / max:{:9.03f}) us'.format(
                t.std(), t.min(), t.max())
        return s

    def to_str(self, show_gpu=False):
        ts = self._ts if show_gpu else self._ts[[0]]
        return '{:<20s}:{}'.format(
            self.name, ' '.join([self._to_str_per_item(t) for t in ts]))

    def __str__(self):
        return self.to_str(show_gpu=True)


def run(name, func, args=(), n=10000, *, n_warmup=10):
    ts = numpy.empty((2, n,), dtype=numpy.float64)
    ev1 = cupy.cuda.stream.Event()
    ev2 = cupy.cuda.stream.Event()

    for i in range(n_warmup):
        func(*args)

    for i in range(n):
        ev1.synchronize()
        ev1.record()
        t1 = time.perf_counter()

        func(*args)

        t2 = time.perf_counter()
        ev2.record()
        ev2.synchronize()
        cpu_time = t2 - t1
        gpu_time = cupy.cuda.get_elapsed_time(ev1, ev2) * 1e-3
        ts[0, i] = cpu_time
        ts[1, i] = gpu_time

    return _PerfCaseResult(name, ts)


@contextlib.contextmanager
def measure(name, show_gpu=False):
    ev1 = cupy.cuda.stream.Event()
    ev2 = cupy.cuda.stream.Event()

    try:
        ev1.synchronize()
        ev1.record()
        t1 = time.perf_counter()
        yield

    finally:
        t2 = time.perf_counter()
        ev2.record()
        ev2.synchronize()
        cpu_time = t2 - t1
        gpu_time = cupy.cuda.get_elapsed_time(ev1, ev2) * 1e-3
        ts = numpy.array([[cpu_time], [gpu_time]])
        perf = _PerfCaseResult(name, ts)
        print(perf.to_str(show_gpu=show_gpu))
