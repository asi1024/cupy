import sys
import time

import cupy
import numpy


class _PerfCaseResult(object):
    def __init__(self, name, ts):
        assert ts.ndim == 2 and ts.shape[0] == 2 and ts.shape[1] > 0
        self.name = name
        self._ts = ts

    @property
    def gpu_times(self):
        return self._ts[1]


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


def main(log_size):

    block_strides = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    dtypes = ['float32', 'int64']

    print('{}\t{}'.format('array_size', 2 ** log_size))
    print('{}\t{}\t{}\t{}\t{}'.format(
        'output_size',
        'reduction_size',
        'dtype',
        'out_contiguous',
        '\t'.join(['block_stride={}'.format(bs) for bs in block_strides])))
    
    for dtype in dtypes:
        dtype = numpy.dtype(dtype)
        for out_contiguous in [False, True]:
            for i in range(0, log_size + 1):
                results = []
                for block_stride in block_strides:
                    cupy.core._kernel.set_block_stride(block_stride)
                    out_size, reduce_size = 2 ** i, 2 ** (log_size - i)
                    if out_contiguous:
                        x = cupy.testing.shaped_random(
                            (reduce_size, out_size), dtype=dtype).T
                    else:
                        x = cupy.testing.shaped_random(
                            (out_size, reduce_size), dtype=dtype)

                    name = 'temp'
                    gpu_times = run(name, cupy.sum, (x, 1), 5, n_warmup=1).gpu_times
                    mean_us = numpy.sort(gpu_times[1:-1]).mean() * 1e6
                    if mean_us < 30000:
                        gpu_times = run(name, cupy.sum, (x, 1), 30, n_warmup=3).gpu_times
                        mean_us = numpy.sort(gpu_times[5:-5]).mean() * 1e6
                    if mean_us < 3000:
                        gpu_times = run(name, cupy.sum, (x, 1), 1000).gpu_times
                        mean_us = numpy.sort(gpu_times[50:-50]).mean() * 1e6
                    results.append('%9.3f' % mean_us)

                print('{}\t{}\t{}\t{}\t{}'.format(
                    out_size,
                    reduce_size,
                    dtype.name,
                    out_contiguous,
                    '\t'.join(results)))
                sys.stdout.flush()

            print('')

with cupy.cuda.Device(0):
    main(16)
    main(20)
    main(24)
