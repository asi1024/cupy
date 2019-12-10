import mock
import unittest

import numpy

import cupy
import cupyx


class TestRun(unittest.TestCase):

    def test_cpu_routine(self):
        with mock.patch('time.perf_counter',
                        mock.Mock(side_effect=[2.4, 3.8] * 10)):
            with mock.patch('cupy.cuda.get_elapsed_time',
                            mock.Mock(return_value=2500)):
                mock_func = mock.Mock()
                x = cupy.testing.shaped_random((2, 3), cupy, 'int32')
                y = cupy.testing.shaped_random((2, 3), cupy, 'int32')
                assert mock_func.call_count == 0

                perf = cupyx.run(
                    'test_name_xxx', mock_func, (x, y), n=10, n_warmup=3)

                assert perf.name == 'test_name_xxx'
                assert mock_func.call_count == 13
                assert (perf.cpu_times == 1.4).all()
                assert (perf.gpu_times == 2.5).all()


class TestMeasure(unittest.TestCase):

    def test_show_gpu(self):
        with mock.patch('time.perf_counter',
                        mock.Mock(side_effect=[2.4, 3.8] * 10)):
            with mock.patch('cupy.cuda.get_elapsed_time',
                            mock.Mock(return_value=2500)):
                with mock.patch('cupyx.time._PerfCaseResult') as m:
                    x = cupy.array([1, 2, 3])
                    y = cupy.array([1, 2, 3])
                    with cupyx.measure('test_name_xxx', show_gpu=True):
                        x + y
                    m.asseert_called_once()
                    name, ts = m.call_args[0]
                    assert name == 'test_name_xxx'
                    assert ts.shape == (2, 1)
                    assert (ts[0] == 1.4).all()
                    assert (ts[1] == 2.5).all()


class TestPerfCaseResult(unittest.TestCase):
    def test_show_gpu(self):
        times = numpy.array([
            [5.4, 7.1, 6.0, 5.4, 4.2],
            [6.4, 4.3, 8.9, 9.6, 3.8],
        ]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times)
        expected = (
            'test_name_xxx       :'
            '     5.620 us   +/- 0.943 (min:    4.200 / max:    7.100) us '
            '     6.600 us   +/- 2.344 (min:    3.800 / max:    9.600) us'
        )
        assert str(perf) == expected

    def test_no_show_gpu(self):
        times = numpy.array([
            [5.4, 7.1, 6.0, 5.4, 4.2],
            [6.4, 4.3, 8.9, 9.6, 3.8],
        ]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times)
        expected = (
            'test_name_xxx       :'
            '     5.620 us   +/- 0.943 (min:    4.200 / max:    7.100) us'
        )
        assert perf.to_str() == expected

    def test_single_show_gpu(self):
        times = numpy.array([[5.4], [6.4]]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times)
        assert str(perf) == 'test_name_xxx       :     5.400 us      6.400 us'

    def test_single_no_show_gpu(self):
        times = numpy.array([[5.4], [6.4]]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times)
        assert perf.to_str() == 'test_name_xxx       :     5.400 us'
